import polars as pl
import numpy as np
import torch
from datetime import datetime
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from os.path import join
from typing import Dict, List, Tuple, Optional

def load_and_prepare_data(config: Dict, verbose: bool = True) -> Data:
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    
    data = Data()

    try:
        node_features = pl.read_parquet(join(config['data_path'], 'node_features.parquet'))
        if 'timestamp' in node_features.columns:
            node_features = node_features.drop(['timestamp'])
        data.x = torch.tensor(node_features.to_numpy(), dtype=torch.float)
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded node features: {data.x.shape}")
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load node features: {e}")

    try:
        edge_features = pl.read_parquet(join(config['data_path'], 'edge_features.parquet'))
        data.edge_attr = torch.tensor(edge_features.to_numpy(), dtype=torch.float)
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded edge features: {data.edge_attr.shape}")
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load edge features: {e}")
    
    try:
        edge_index = pl.read_parquet(join(config['data_path'], 'edge_index.parquet')).to_numpy().T
        data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded edge index: {data.edge_index.shape}")
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load edge index: {e}")
    
    # Load labels
    try:
        label = pl.read_parquet(join(config['data_path'], f"label_{config['mode']}.parquet"))
        label = label.to_numpy().reshape(-1)
        data.edge_label = torch.tensor(label, dtype=torch.long)
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded labels: {data.edge_label.shape}")
    except Exception as e:
        raise RuntimeError(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to load labels: {e}")
    
    num_edges = data.num_edges
    indices = np.arange(num_edges)
    test_size = int(len(indices) * config['test_size'])
    val_size = int(len(indices) * config['valid_size'])
    
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=config.get('seed', 42), 
        stratify=data.edge_label
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=config.get('seed', 42),
        stratify=data.edge_label[train_idx]
    )
    
    # Create masks
    data.train_mask = torch.zeros(num_edges, dtype=torch.bool)
    data.val_mask = torch.zeros(num_edges, dtype=torch.bool)
    data.test_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    data.train_idx = torch.tensor(train_idx, dtype=torch.long)
    data.val_idx = torch.tensor(val_idx, dtype=torch.long)
    data.test_idx = torch.tensor(test_idx, dtype=torch.long)
    
    if config.get('inductive', True):
        data.train_edge_index = data.edge_index[:, data.train_mask]
        data.train_edge_attr = data.edge_attr[data.train_mask]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Inductive mode: Using only train edges for message passing")
    
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data split:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Total nodes: {num_edges:,}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Train: {len(train_idx):,} ({len(train_idx)/num_edges*100:.1f}%)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Val: {len(val_idx):,} ({len(val_idx)/num_edges*100:.1f}%)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Test: {len(test_idx):,} ({len(test_idx)/num_edges*100:.1f}%)")
    
    # Compute class weights if needed
    if config.get('use_class_weight', False):
        train_labels = data.edge_label[train_idx]
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        num_classes = len(unique_classes)
        
        # Inverse frequency weighting
        total_samples = len(train_labels)
        class_weight = total_samples / (num_classes * class_counts)
        class_weight = np.sqrt(class_weight)
        class_weight = class_weight / class_weight.sum() * num_classes
        
        data.class_weight = class_weight
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Class distribution (training set):")
            for cls in range(num_classes):
                count = class_counts[cls]
                weight = class_weight[cls]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Class {cls}: {count:,} samples ({count/total_samples*100:.2f}%), weight: {weight:.4f}")
    
    return data
