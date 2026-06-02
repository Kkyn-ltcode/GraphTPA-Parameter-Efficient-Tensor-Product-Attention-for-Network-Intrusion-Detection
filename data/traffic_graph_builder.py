import os
import sys
import json
import torch
import polars as pl
import argparse

from os.path import join
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from datetime import datetime

from analyze_data import analyze_netflow_data
from optimize_dataframe import optimize_dataframe
from build_node_features import build_node_features

numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int128, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]

def parse_args():
    parser = argparse.ArgumentParser(description='Building Traffic Graph')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True
    )
    return parser.parse_args()

class TrafficGraphBuilder():
    def __init__(self, config):
        self.traffic_folder = join(config['data_path'], 'traffic_graph')
        self.numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int128, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        os.makedirs(self.traffic_folder, exist_ok=True)

        filename = os.path.basename(config['data_path'])
        self.data_path = join(config['data_path'], filename + '.csv')
        self.analysis_path = join(self.traffic_folder, filename + '.json')

        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading data from: {self.data_path}")
            self.df = pl.read_csv(self.data_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded {len(self.df):,} records")
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
        self.preprocessing()

    def preprocessing(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data cleaning...")
        original_count = len(self.df)
        self.df = self.df.unique()
        removed_dupes = original_count - len(self.df)
        print(f"  → Removed {removed_dupes:,} duplicate rows ({removed_dupes/original_count*100:.2f}%)")
    
        self.df = self.df.with_columns([
            (pl.col("IPV4_SRC_ADDR") + ":" + pl.col("L4_SRC_PORT").cast(pl.String)).alias("src_node"),
            (pl.col("IPV4_DST_ADDR") + ":" + pl.col("L4_DST_PORT").cast(pl.String)).alias("dst_node"),
        ])
        node_map = pl.concat([
            self.df.select('src_node').rename({'src_node': 'node'}), 
            self.df.select('dst_node').rename({'dst_node': 'node'})
        ]).unique().sort(by='node').with_row_index('node_id')
        self.num_nodes = len(node_map)
        self.df = self.df.join(node_map, left_on='src_node', right_on='node', how='left').rename({'node_id': 'src_node_id'})
        self.df = self.df.join(node_map, left_on='dst_node', right_on='node', how='left').rename({'node_id': 'dst_node_id'})
        self.df = self.df.with_row_index('flow_id')
        
    def analyze_data(self):
        analyze_netflow_data(self.df, self.analysis_path)

    def normalize_feature(self, df):
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input must be a Polars DataFrame")
        
        normalized_df = df.clone()
        col_to_drop = []
        
        for col in df.columns:
            # Skip non-numeric columns
            if df[col].dtype not in self.numeric_types:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {col}: skipped (non-numeric)")
                continue
            
            # Calculate statistics
            col_data = df[col]
            col_min = col_data.min()
            col_max = col_data.max()
            col_mean = col_data.mean()
            col_std = col_data.std()
            zeros_count = (col_data == 0).sum()
            zeros_pct = zeros_count / len(col_data) if len(col_data) > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {col}: min={col_min:.2e}, max={col_max:.2e}, mean={col_mean:.2e}, std={col_std:.2e}, zeros={(zeros_pct * 100):.1f}%")
            
            # Skip constant features
            if col_std is None or col_std < 1e-10:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {col}: constant → zero → drop")
                normalized_df = normalized_df.with_columns(pl.lit(0.0).alias(col))
                col_to_drop.append(col)
                continue
            
            # For very sparse features (>80% zeros)
            if zeros_pct > 0.8:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  → binary")
                normalized_df = normalized_df.with_columns(
                    (pl.col(col) > 0).cast(pl.Float64).alias(col)
                )
                continue
            
            # For positive features with large range
            if col_min is not None and col_min >= 0 and col_max / (col_std + 1e-10) > 100:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  → log+scale")
                normalized_df = normalized_df.with_columns(
                    pl.col(col).log1p().alias(f"{col}_log")
                )
                log_mean = normalized_df[f"{col}_log"].mean()
                log_std = normalized_df[f"{col}_log"].std()
                
                normalized_df = normalized_df.with_columns(
                    ((pl.col(f"{col}_log") - log_mean) / (log_std + 1e-8))
                    .clip(-5, 5)
                    .alias(col)
                ).drop(f"{col}_log")
            else:
                # Standard normalization
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  → standard scaling")
                normalized_df = normalized_df.with_columns(
                    ((pl.col(col) - col_mean) / (col_std + 1e-8))
                    .clip(-5, 5)
                    .alias(col)
                )
        
        # Calculate final statistics
        df_numpy = normalized_df.to_numpy()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After normalize:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Min={df_numpy.min():.2e}, Max={df_numpy.max():.2e}, Mean={df_numpy.mean():.2e}, Std={df_numpy.std():.2e}")
        
        return normalized_df.drop(col_to_drop)

    def preprocessing_feature(self, df):
        df = df.select([
            pl.when(pl.col(col).is_infinite()).then(float('nan')).otherwise(pl.col(col)).alias(col) for col in df.columns
        ])
        data_numpy = df.to_numpy()
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data_numpy)

        feature_matrix = torch.tensor(data_imputed)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Original stats:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Min: {feature_matrix.min():.2e}, Max: {feature_matrix.max():.2e}, Mean: {feature_matrix.mean():.2e}, Std: {feature_matrix.std():.2e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Clipping catastrophic values...")
        feature_matrix = torch.clamp(feature_matrix, min=-1e10, max=1e10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After clipping:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Min: {feature_matrix.min():.2e}, Max: {feature_matrix.max():.2e}, Mean: {feature_matrix.mean():.2e}, Std: {feature_matrix.std():.2e}")
            
        df_imputed = df.with_columns([
            pl.Series(name=col, values=feature_matrix[:, i])
            for i, col in enumerate(df.columns)
        ])
        return df_imputed
    
    def create_edge_features(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Start building traffic edge features...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Normalizing edge features...")
        cols_to_drop = []
        for col in ['L4_SRC_PORT', 'L4_DST_PORT', 'Label', 'flow_id', 'src_node_id', 'dst_node_id', 'label_multi']:
            if col in self.df.columns:
                cols_to_drop.append(col)
        df = self.df.drop(cols_to_drop)
        num_cols = [col for col in df.columns if df[col].dtype in self.numeric_types]
        df = df[num_cols]

        df = self.preprocessing_feature(df)
        df_norm = self.normalize_feature(df)
        df_op = optimize_dataframe(df_norm, verbose=False)
        
        save_path = join(self.traffic_folder, 'edge_features.parquet')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving edge features to {save_path}")
        df_op.write_parquet(save_path)

    def create_node_feautures(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating node features...")
        df = build_node_features(self.df)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Normalizing node features...")
        num_cols = [col for col in df.columns if df[col].dtype in self.numeric_types]
        df = df[num_cols]

        df = self.preprocessing_feature(df)
        df_norm = self.normalize_feature(df)
        df_op = optimize_dataframe(df_norm, verbose=False)
        
        save_path = join(self.traffic_folder, 'node_features.parquet')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving node features to {save_path}")
        df_op.write_parquet(save_path)

    def create_edge_index(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating edge index...")
        edge_index = self.df.select(['src_node_id', 'dst_node_id']).with_columns(pl.all().cast(pl.UInt32))
        save_path = join(self.traffic_folder, 'edge_index.parquet')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving edge index to {save_path}")
        edge_index.write_parquet(save_path)
    
    def generate_label(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating Label...")
        label2id = {label: idx for idx, label in enumerate(self.df['Attack'].value_counts().sort(by='count', descending=True)['Attack'])}
        id2label = {v: k for k, v in label2id.items()}
        label_df = pl.DataFrame({
            'label': list(label2id.keys()),
            'value': list(label2id.values())
        })
        self.df = self.df.join(label_df, left_on='Attack', right_on='label', how='left').rename({'value': 'label_multi'})

        save_path = join(self.traffic_folder, 'label_multi.parquet')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving multi label to {save_path}")
        self.df.select('label_multi').with_columns(pl.all().cast(pl.UInt8)).write_parquet(save_path)
        
        save_path = join(self.traffic_folder, 'label_binary.parquet')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving binary label to {save_path}")
        self.df.select('Label').with_columns(pl.all().cast(pl.UInt8)).write_parquet(save_path)

        with open(join(self.traffic_folder, 'id2label_multi.json'), 'w', encoding='utf-8') as f:
            json.dump(id2label, f, indent=2)
            
        with open(join(self.traffic_folder, 'id2label_binary.json'), 'w', encoding='utf-8') as f:
            json.dump({"0": "Benign", "1": "Attack"}, f, indent=2)

def main():
    args = parse_args()
    config = json.load(open(args.config_path, 'r', encoding='utf-8'))
    builder = TrafficGraphBuilder(config)
    if config['traffic']['analyze_data']:
        builder.analyze_data()
    builder.create_edge_features()
    builder.create_node_feautures()
    builder.create_edge_index()
    builder.generate_label()

if __name__ == '__main__':
    main()
