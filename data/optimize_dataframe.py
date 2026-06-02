import polars as pl
from datetime import datetime

def optimize_dataframe(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Optimizing dataframe...")
    original_memory = df.estimated_size("mb")
    
    optimized_cols = []
    
    for col in df.columns:
        dtype = df[col].dtype
        series = df[col]
        
        # Handle integer types
        if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            # Get min and max values, handling nulls
            col_min = series.min()
            col_max = series.max()
            
            if col_min is not None and col_max is not None:
                # Determine smallest integer type that can hold the data
                if col_min >= 0:  # Unsigned integers
                    if col_max <= 255:
                        optimized_cols.append(series.cast(pl.UInt8))
                    elif col_max <= 65535:
                        optimized_cols.append(series.cast(pl.UInt16))
                    elif col_max <= 4294967295:
                        optimized_cols.append(series.cast(pl.UInt32))
                    else:
                        optimized_cols.append(series.cast(pl.UInt64))
                else:  # Signed integers
                    if col_min >= -128 and col_max <= 127:
                        optimized_cols.append(series.cast(pl.Int8))
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_cols.append(series.cast(pl.Int16))
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_cols.append(series.cast(pl.Int32))
                    else:
                        optimized_cols.append(series)
            else:
                optimized_cols.append(series)
        
        # Handle unsigned integer types
        elif dtype in [pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
            col_max = series.max()
            
            if col_max is not None:
                if col_max <= 255:
                    optimized_cols.append(series.cast(pl.UInt8))
                elif col_max <= 65535:
                    optimized_cols.append(series.cast(pl.UInt16))
                elif col_max <= 4294967295:
                    optimized_cols.append(series.cast(pl.UInt32))
                else:
                    optimized_cols.append(series)
            else:
                optimized_cols.append(series)
        
        # Handle float types
        elif dtype == pl.Float64:
            # Try to downcast to Float32 if precision allows
            # Check if all values can be represented in Float32
            optimized_cols.append(series.cast(pl.Float32))
        
        # Handle string types - convert to categorical if cardinality is low
        elif dtype == pl.Utf8:
            unique_ratio = series.n_unique() / len(series)
            
            # If less than 50% unique values, use categorical
            if unique_ratio < 0.5:
                optimized_cols.append(series.cast(pl.Categorical))
            else:
                optimized_cols.append(series)
        
        # Keep other types as-is
        else:
            optimized_cols.append(series)
    
    # Create optimized DataFrame
    optimized_df = pl.DataFrame(optimized_cols)
    
    # Calculate and display memory savings
    optimized_memory = optimized_df.estimated_size("mb")
    memory_saved = original_memory - optimized_memory
    percent_saved = (memory_saved / original_memory) * 100
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Original memory usage: {original_memory:.2f} MB")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Optimized memory usage: {optimized_memory:.2f} MB")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Memory saved: {memory_saved:.2f} MB ({percent_saved:.1f}%)")
    if verbose: 
        print(f"\nData type changes:")
        for col in df.columns:
            old_dtype = df[col].dtype
            new_dtype = optimized_df[col].dtype
            if old_dtype != new_dtype:
                print(f"  {col}: {old_dtype} -> {new_dtype}")
    
    return optimized_df
