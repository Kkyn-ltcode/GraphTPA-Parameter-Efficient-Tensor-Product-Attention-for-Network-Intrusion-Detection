import os
import gc
import wandb
import json
import math
import torch
import time
import warnings
import numpy as np
import polars as pl
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from utils import process_report, set_seed
from traffic_data_loader import load_and_prepare_data
from traffic_graph_model import GraphTransformer
from label_smoothing import LabelSmoothingCrossEntropy
from focal_loss import FocalLoss
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Multi Graph Transformer for Network Traffic Classification")
    
    parser.add_argument('--config_path', type=str, required=True, help='Path to config JSON file')
    return parser.parse_args()

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train_epoch(
    model: nn.Module,
    loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
    epoch: int,
    scaler: GradScaler,
    id2label: Dict,
    dataset: str,
    criterion: nn.Module,
    edge_attr: torch.tensor,
    grad_clip: float = 1.0,
    accumulation_steps: int = 1
) -> Tuple[float, Dict, str]:
    
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    num_classes = loader.data.edge_label.unique().size(0)
    
    if rank == 0:
        loader = tqdm(loader, desc=f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} [Train] [{dataset}]")
        
    for batch_idx, batch in enumerate(loader):
        batch_edge_attr = edge_attr[batch.input_id].to(device, non_blocking=True) if edge_attr is not None else None
        batch = batch.to(device, non_blocking=True)
        batch_label = batch.edge_label
        
        with autocast('cuda'):
            _, out = model(
                batch.x, 
                batch.edge_index,
                batch.edge_attr,
                batch_edge_attr,
                batch.edge_label_index
            )
            loss = criterion(out, batch_label)
            
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                    
        with torch.no_grad():
            pred = out.argmax(dim=1)
            total_loss += loss.item() * len(batch_label) * accumulation_steps
            total_samples += len(batch_label)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    report = process_report(
        classification_report(
            all_labels, all_preds,
            zero_division=0,
            target_names=[id2label[str(i)] for i in range(num_classes)],
            output_dict=True
        )
    )
    
    report_table = classification_report(
        all_labels, all_preds,
        zero_division=0,
        digits=4,
        target_names=[id2label[str(i)] for i in range(num_classes)]
    )
    return avg_loss, report, report_table

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: LinkNeighborLoader,
    device: torch.device,
    rank: int,
    id2label: Dict,
    criterion: nn.Module,
    edge_attr: torch.tensor,
    mask_name: str = 'Val'
) -> Tuple[float, Dict, str]:
    
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    edge_embs = []
    num_classes = loader.data.edge_label.unique().size(0)
    
    if rank == 0:
        loader = tqdm(loader, desc=f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating ({mask_name})")

    for batch in loader:
        batch_edge_attr = edge_attr[batch.input_id].to(device, non_blocking=True) if edge_attr is not None else None
        batch = batch.to(device, non_blocking=True)
        batch_label = batch.edge_label
        
        with autocast('cuda'):
            edge_emb, out = model(
                batch.x, 
                batch.edge_index,
                batch.edge_attr,
                batch_edge_attr,
                batch.edge_label_index
            )
            loss = criterion(out, batch_label)
            
        pred = out.argmax(dim=1)
        total_samples += len(batch_label)
        total_loss += loss.item() * len(batch_label)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch_label.cpu().numpy())
        edge_embs.append(edge_emb.detach().cpu())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    report = process_report(
        classification_report(
            all_labels, all_preds,
            zero_division=0,
            target_names=[id2label[str(i)] for i in range(num_classes)],
            output_dict=True
        )
    )
    
    report_table = classification_report(
        all_labels, all_preds,
        zero_division=0,
        digits=4,
        target_names=[id2label[str(i)] for i in range(num_classes)]
    )
    return avg_loss, report, report_table

def train_worker(rank: int, world_size: int, data: Data, config: Dict):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    set_seed(config.get('seed', 42) + rank)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if is_main_process():
        train_start = time.time()
        
    if rank == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating Data Loader")

    class_weight = data.class_weight if hasattr(data, 'class_weight') else None

    use_inductive = config.get('inductive', True) and hasattr(data, 'train_edge_index')
    
    if use_inductive:
        # Create a copy of data with only training edges for message passing
        train_data = Data(
            x=data.x,
            edge_index=data.train_edge_index,  # Only train edges for neighbor sampling
            edge_attr=data.train_edge_attr,
            edge_label=data.edge_label,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
        )
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Using inductive split: message passing only on training edges")
    else:
        train_data = data
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Using transductive split: message passing on all edges")
            
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=config['num_neighbors'],
        batch_size=config['batch_size'],
        shuffle=True,
        edge_label_index=data.edge_index[:, data.train_mask],
        edge_label=data.edge_label[data.train_mask],
        num_workers=config['num_workers'],
        persistent_workers=True,
        prefetch_factor=config['prefetch_factor'],
        drop_last=True,
        neg_sampling=None,
    )
    
    val_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=config['num_neighbors'],
        batch_size=config['batch_size'],
        shuffle=False,
        edge_label_index=data.edge_index[:, data.val_mask],
        edge_label=data.edge_label[data.val_mask],
        num_workers=config['num_workers'],
        persistent_workers=True,
        prefetch_factor=config['prefetch_factor'],
    )

    test_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=config['num_neighbors'],
        batch_size=config['batch_size'],
        shuffle=False,
        edge_label_index=data.edge_index[:, data.test_mask],
        edge_label=data.edge_label[data.test_mask],
        num_workers=config['num_workers'],
        persistent_workers=True,
        prefetch_factor=config['prefetch_factor'],
    )

    if rank == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing model")
    
    model = GraphTransformer(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=config['hidden_channels'],
        out_channels=config['num_classes'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_edge_features=config['use_edge_features'],
        useTrans=config['useTrans'],
        useGCN=config['useGCN'],
        useSAGE=config['useSAGE'],
        use_tpa=config.get('use_tpa', True),
        use_rope=config.get('use_rope', False),
        rank_adaptation=config.get('rank_adaptation', 'hierarchical'),
        use_tensor_edge=config.get('use_tensor_edge', True),
        tensor_edge_rank=config.get('tensor_edge_rank', None),
        tensor_edge_mode=config.get('tensor_edge_mode', 'hybrid'),
        activation=config.get('activation', 'gelu')
    ).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Total parameters: {total_params:,}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trainable parameters: {trainable_params:,}")

    # if config['useGCN'] or config['useSAGE']:
    #     model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    # else:
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        
    label_smoothing = config.get('label_smoothing', 0.1)
    
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=class_weight if class_weight is not None else None,
            gamma=config.get('focal_gamma', 2.0),
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Focal Loss (gamma={config['focal_gamma']}, label_smoothing={label_smoothing})")
    elif config.get('use_label_smoothing', True):
        criterion = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weight,
            reduction='mean'
        )
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Label Smoothing Cross Entropy (smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Using Cross Entropy Loss")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=config.get('warmup_factor', 0.1),
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - warmup_epochs,
        eta_min=config.get('min_lr', 1e-6)
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    scaler = GradScaler(
        init_scale=2**10,
        growth_factor=1.5,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )

    if rank == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training on {world_size} GPUs")
    
    best_val_f1 = 0
    patience_counter = 0
    results = {'train': [], 'val': []}
    train_results = []
    val_results = []

    try:
        with open(join(config['data_path'], f"id2label_{config['mode']}.json"), 'r', encoding='utf-8') as f:
            id2label = json.load(f)
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load id2label mapping: {e}")

    if rank == 0:
        with open(join(config['output_path'], 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
    dataset = config['data_path'].split('/')[-2]
    for epoch in range(config['epochs']):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main_process():
            epoch_start = time.time()
        
        if rank == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}/{config['epochs']}")

        train_loss, train_report, train_metrics = train_epoch(
            model, train_loader, optimizer, device, rank, epoch,
            scaler, id2label, dataset, criterion, data.edge_attr[data.train_mask],
            grad_clip=config.get('grad_clip', 1.0),
            accumulation_steps=config.get('accumulation_steps', 1)
        )

        val_loss, val_report, val_metrics = evaluate(
            model, val_loader, device, rank, id2label, criterion, data.edge_attr[data.val_mask], 'Val'
        )
        
        scheduler.step()
        
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Learning Rate: {current_lr}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Train F1 (macro): {train_report['macro avg']['f1-score']:.2f}%")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Val F1 (macro): {val_report['macro avg']['f1-score']:.2f}%")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Train report")
            print(train_metrics)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Val report")
            print(val_metrics)
            
            results['train'].append(train_report)
            results['val'].append(val_report)
            
            current_val_f1 = val_report['macro avg']['f1-score']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1': current_val_f1,
                    'config': config
                }
                torch.save(checkpoint, join(config['output_path'], 'model.pt'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved best model (F1: {best_val_f1:.2f}%)")
            else:
                patience_counter += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No improvement ({patience_counter}/{config.get('patience', 10)})")

            with open(join(config['output_path'], f"metrics.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            if patience_counter >= config.get('patience', 10):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Early stopping triggered after {epoch + 1} epochs")
                break

        if (epoch + 1) % 1 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if is_main_process():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch Time: {time.time() - epoch_start:.2f}s")

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    if is_main_process():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Total Training Time: {time.time() - train_start:.2f}s")
            
    if is_main_process():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Final Evaluation on Test Set")
        
    # All ranks must load the best checkpoint to evaluate correctly, or only rank 0 evaluates
    if dist.is_initialized():
        dist.barrier()
    
    # Actually, we only need to evaluate on rank 0 if we aren't using DistributedSampler
    # But since all ranks call evaluate(), all ranks should load it.
    try:
        checkpoint = torch.load(join(config['output_path'], f"model.pt"), weights_only=False, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        if is_main_process():
            print(f"Could not load best model: {e}")
        
    test_loss, test_report, test_metrics = evaluate(
        model, test_loader, device, rank, id2label, criterion, data.edge_attr[data.test_mask], 'Test'
    )

    if is_main_process():
        results['test'] = test_report
        results['total_training_time'] = time.time() - train_start

    if rank == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Test Loss: {test_loss:.4f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Test F1 (macro): {test_report['macro avg']['f1-score']:.2f}%")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Classification Report:")
        print(test_metrics)

        with open(join(config['output_path'], f"metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    cleanup_ddp()

def main():
    args = parse_args()
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load config from {args.config_path}: {e}")
    set_seed(config.get('seed', 42))
    config['batch_size'] = int(config['base_batch'] * config['num_mul'])

    lr_scale_method = config.get('lr_scale_method', 'sqrt')  # 'sqrt', 'linear', or 'none'
    if lr_scale_method == 'sqrt':
        scaled_lr = config['base_lr'] * math.sqrt(config['num_mul'])
    elif lr_scale_method == 'linear':
        scaled_lr = config['base_lr'] * (config['num_mul'])
    else:
        scaled_lr = config['base_lr']
    config['learning_rate'] = scaled_lr

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch size: {config['batch_size']}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Learning rate: {config['learning_rate']:.6f}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Random seed: {config['seed']}")

    result_folder = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%d-%m-%Y')
    result_step = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%H:%M:%S')
    config['output_path'] = join(config['data_path'], 'results', config['mode'], result_folder, result_step)
    os.makedirs(config['output_path'], exist_ok=True)
    
    data = load_and_prepare_data(config)
    config['num_classes'] = data.edge_label.unique().size(0)
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] No CUDA GPUs available!")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {world_size} GPU(s)")
    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(world_size, data, config),
            nprocs=world_size,
            join=True
        )
    else:
        train_worker(0, 1, data, config)

if __name__ == "__main__":
    main()
