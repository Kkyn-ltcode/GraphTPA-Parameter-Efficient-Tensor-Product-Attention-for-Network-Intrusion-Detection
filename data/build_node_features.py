import polars as pl
from datetime import datetime

numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int128, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]

def has_column(df: pl.DataFrame, col: str) -> bool:
    """Check if a column exists in the dataframe"""
    return col in df.columns

def build_node_features(df: pl.DataFrame) -> pl.DataFrame:
    if not has_column(df, 'src_node_id') or not has_column(df, 'dst_node_id'):
        raise ValueError(f"[{datetime.now().strftime('%H:%M:%S')}] DataFrame must contain 'src_node_id' and 'dst_node_id' columns")
    
    # Collect all unique node IDs
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting node IDs...")
    all_nodes = pl.concat([
        df.select(pl.col('src_node_id').alias('node_id')),
        df.select(pl.col('dst_node_id').alias('node_id'))
    ]).unique().sort('node_id')
    
    # Flow counts
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Flow counts...")
    src_counts = df.group_by('src_node_id').agg(
        pl.len().alias('total_flows_as_source')
    ).rename({'src_node_id': 'node_id'})
    
    dst_counts = df.group_by('dst_node_id').agg(
        pl.len().alias('total_flows_as_destination')
    ).rename({'dst_node_id': 'node_id'})
    
    node_features = all_nodes.join(src_counts, on='node_id', how='left').join(
        dst_counts, on='node_id', how='left'
    ).fill_null(0)
    
    # Traffic volume
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Traffic volume...")
    for col in ['OUT_BYTES', 'OUT_PKTS', 'IN_BYTES', 'IN_PKTS']:
        if has_column(df, col):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] → Processing {col}...")
            src_agg = df.group_by('src_node_id').agg([
                pl.col(col).sum().alias(f'{col.lower()}_as_source_sum'),
                pl.col(col).mean().alias(f'{col.lower()}_as_source_mean'),
                pl.col(col).std().alias(f'{col.lower()}_as_source_std'),
                pl.col(col).max().alias(f'{col.lower()}_as_source_max'),
                pl.col(col).min().alias(f'{col.lower()}_as_source_min'),
                pl.col(col).median().alias(f'{col.lower()}_as_source_median'),
                pl.col(col).quantile(0.95).alias(f'{col.lower()}_as_source_p95')
            ]).rename({'src_node_id': 'node_id'})
            
            dst_agg = df.group_by('dst_node_id').agg([
                pl.col(col).sum().alias(f'{col.lower()}_as_destination_sum'),
                pl.col(col).mean().alias(f'{col.lower()}_as_destination_mean'),
                pl.col(col).std().alias(f'{col.lower()}_as_destination_std'),
                pl.col(col).max().alias(f'{col.lower()}_as_destination_max'),
                pl.col(col).min().alias(f'{col.lower()}_as_destination_min'),
                pl.col(col).median().alias(f'{col.lower()}_as_destination_median'),
                pl.col(col).quantile(0.95).alias(f'{col.lower()}_as_destination_p95')
            ]).rename({'dst_node_id': 'node_id'})
            
            node_features = node_features.join(src_agg, on='node_id', how='left').join(
                dst_agg, on='node_id', how='left'
            )
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] → Skipped: {col} column not found")
    
    # Unique contacts
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Unique contacts...")
    if has_column(df, 'src_node_id') and has_column(df, 'dst_node_id'):
        dst_unique = df.group_by('src_node_id').agg(
            pl.col('dst_node_id').n_unique().alias('unique_destinations_contacted')
        ).rename({'src_node_id': 'node_id'})
    
        src_unique = df.group_by('dst_node_id').agg(
            pl.col('src_node_id').n_unique().alias('unique_sources_contacted')
        ).rename({'dst_node_id': 'node_id'})
    
        node_features = node_features.join(dst_unique, on='node_id', how='left').join(
            src_unique, on='node_id', how='left'
        )
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipped: Required endpoint columns not found")
    
    # Protocol diversity
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Protocol diversity...")
    if has_column(df, 'PROTOCOL'):
        proto_unique_src = df.group_by('src_node_id').agg(
            pl.col('PROTOCOL').n_unique().alias('protocol_diversity_as_source')
        ).rename({'src_node_id': 'node_id'})
        
        proto_unique_dst = df.group_by('dst_node_id').agg(
            pl.col('PROTOCOL').n_unique().alias('protocol_diversity_as_destination')
        ).rename({'dst_node_id': 'node_id'})
        
        node_features = node_features.join(proto_unique_src, on='node_id', how='left').join(
            proto_unique_dst, on='node_id', how='left'
        )
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipped: PROTOCOL column not found")
    
    # Temporal features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Temporal features...")
    if has_column(df, 'FLOW_DURATION_MILLISECONDS'):
        src_duration = df.group_by('src_node_id').agg([
            pl.col('FLOW_DURATION_MILLISECONDS').mean().alias('avg_flow_duration_as_source'),
            pl.col('FLOW_DURATION_MILLISECONDS').std().alias('std_flow_duration_as_source'),
            pl.col('FLOW_DURATION_MILLISECONDS').max().alias('max_flow_duration_as_source'),
            pl.col('FLOW_DURATION_MILLISECONDS').quantile(0.95).alias('p95_flow_duration_as_source')
        ]).rename({'src_node_id': 'node_id'})
        
        dst_duration = df.group_by('dst_node_id').agg([
            pl.col('FLOW_DURATION_MILLISECONDS').mean().alias('avg_flow_duration_as_destination'),
            pl.col('FLOW_DURATION_MILLISECONDS').std().alias('std_flow_duration_as_destination'),
            pl.col('FLOW_DURATION_MILLISECONDS').max().alias('max_flow_duration_as_destination'),
            pl.col('FLOW_DURATION_MILLISECONDS').quantile(0.95).alias('p95_flow_duration_as_destination')
        ]).rename({'dst_node_id': 'node_id'})
        
        node_features = node_features.join(src_duration, on='node_id', how='left').join(
            dst_duration, on='node_id', how='left'
        )
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipped: FLOW_DURATION_MILLISECONDS column not found")
    
    # Coefficient of variation for bytes (measure of variability)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing coefficient of variation...")
    cv_features = []
    if has_column(node_features, 'in_bytes_as_source_mean') and has_column(node_features, 'in_bytes_as_source_std'):
        cv_features.append(
            (pl.col('in_bytes_as_source_std') / (pl.col('in_bytes_as_source_mean') + 1)).alias('in_bytes_as_source_cv')
        )
    if has_column(node_features, 'in_bytes_as_destination_mean') and has_column(node_features, 'in_bytes_as_destination_std'):
        cv_features.append(
            (pl.col('in_bytes_as_destination_std') / (pl.col('in_bytes_as_destination_mean') + 1)).alias('in_bytes_as_destination_cv')
        )
    if has_column(node_features, 'out_bytes_as_source_mean') and has_column(node_features, 'out_bytes_as_source_std'):
        cv_features.append(
            (pl.col('out_bytes_as_source_std') / (pl.col('out_bytes_as_source_mean') + 1)).alias('out_bytes_as_source_cv')
        )
    if has_column(node_features, 'out_bytes_as_destination_mean') and has_column(node_features, 'out_bytes_as_destination_std'):
        cv_features.append(
            (pl.col('out_bytes_as_destination_std') / (pl.col('out_bytes_as_destination_mean') + 1)).alias('out_bytes_as_destination_cv')
        )
    
    if cv_features:
        node_features = node_features.with_columns(cv_features)
    
    # Combined/derived features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing combined features...")
    combined_features = []
    
    # Total flows (source + destination)
    if has_column(node_features, 'total_flows_as_source') and has_column(node_features, 'total_flows_as_destination'):
        combined_features.extend([
            (pl.col('total_flows_as_source') + pl.col('total_flows_as_destination')).alias('total_flows'),
            (pl.col('total_flows_as_source') / (pl.col('total_flows_as_source') + pl.col('total_flows_as_destination') + 1)).alias('source_flow_ratio'),
            (pl.col('total_flows_as_destination') / (pl.col('total_flows_as_source') + pl.col('total_flows_as_destination') + 1)).alias('destination_flow_ratio')
        ])
    
    # Fan-in and fan-out (network connectivity)
    if has_column(node_features, 'unique_destinations_contacted') and has_column(node_features, 'unique_sources_contacted'):
        combined_features.extend([
            pl.col('unique_destinations_contacted').alias('fan_out'),
            pl.col('unique_sources_contacted').alias('fan_in'),
            (pl.col('unique_destinations_contacted') + pl.col('unique_sources_contacted')).alias('degree'),
            (pl.col('unique_destinations_contacted') + pl.col('unique_sources_contacted')).alias('total_unique_peers')
        ])
    
    # Directional behavior flags
    if has_column(node_features, 'total_flows_as_source') and has_column(node_features, 'total_flows_as_destination'):
        combined_features.extend([
            (pl.col('total_flows_as_source') > pl.col('total_flows_as_destination') * 4).cast(pl.Int8).alias('is_primarily_source'),
            (pl.col('total_flows_as_destination') > pl.col('total_flows_as_source') * 4).cast(pl.Int8).alias('is_primarily_destination'),
            (
                (pl.col('total_flows_as_source') > 0) & 
                (pl.col('total_flows_as_destination') > 0) &
                (pl.col('total_flows_as_source') / (pl.col('total_flows_as_destination') + 1) < 4) &
                (pl.col('total_flows_as_destination') / (pl.col('total_flows_as_source') + 1) < 4)
            ).cast(pl.Int8).alias('is_bidirectional_node')
        ])
    
    # Total bytes (if available)
    if has_column(node_features, 'in_bytes_as_source_sum') and has_column(node_features, 'in_bytes_as_destination_sum'):
        combined_features.append(
            (pl.col('in_bytes_as_source_sum') + pl.col('in_bytes_as_destination_sum')).alias('total_bytes_in')
        )
    if has_column(node_features, 'out_bytes_as_source_sum') and has_column(node_features, 'out_bytes_as_destination_sum'):
        combined_features.append(
            (pl.col('out_bytes_as_source_sum') + pl.col('out_bytes_as_destination_sum')).alias('total_bytes_out')
        )
    
    # Total packets (if available)
    if has_column(node_features, 'in_pkts_as_source_sum') and has_column(node_features, 'in_pkts_as_destination_sum'):
        combined_features.append(
            (pl.col('in_pkts_as_source_sum') + pl.col('in_pkts_as_destination_sum')).alias('total_pkts_in')
        )
    if has_column(node_features, 'out_pkts_as_source_sum') and has_column(node_features, 'out_pkts_as_destination_sum'):
        combined_features.append(
            (pl.col('out_pkts_as_source_sum') + pl.col('out_pkts_as_destination_sum')).alias('total_pkts_out')
        )
    
    if combined_features:
        node_features = node_features.with_columns(combined_features)
    
    # Fill any remaining nulls with 0
    node_features = node_features.fill_null(0)
    return node_features.drop(['node_id'])
