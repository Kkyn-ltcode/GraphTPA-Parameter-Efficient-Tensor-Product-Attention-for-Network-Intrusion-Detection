import polars as pl
import json
import sys
from datetime import datetime

numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int128, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]

def analyze_netflow_data(df: pl.DataFrame, output_file: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting dataset analysis...")
    
    results = {}
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting basic dataset information...")
    
    results["dataset_overview"] = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "feature_names": df.columns,
        "memory_usage_mb": round(df.estimated_size("mb"), 2),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing data types and schema...")
    
    results["schema_info"] = {
        col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing missing values...")
    
    null_counts = df.null_count()
    missing_analysis = {}
    
    for col in df.columns:
        null_count = null_counts[col][0]
        missing_analysis[col] = {
            "null_count": null_count,
            "null_percentage": round((null_count / len(df)) * 100, 4)
        }
    
    results["missing_values"] = missing_analysis
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing label distribution...")
    
    if "Attack" in df.columns:
        label_dist = df.group_by("Attack").agg(pl.count().alias("count")).sort("count", descending=True)
        
        results["attack_distribution"] = {
            "unique_vs": len(label_dist),
            "attack_counts": {
                row["Attack"]: int(row["count"]) 
                for row in label_dist.iter_rows(named=True)
            },
            "attack_percentages": {
                row["Attack"]: round((row["count"] / len(df)) * 100, 4)
                for row in label_dist.iter_rows(named=True)
            }
        }
    
    if "Label" in df.columns:
        attack_dist = df.group_by("Label").agg(pl.count().alias("count"))
        
        results["label_distribution"] = {
            row["Label"]: {
                "count": int(row["count"]),
                "percentage": round((row["count"] / len(df)) * 100, 4)
            }
            for row in attack_dist.iter_rows(named=True)
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing protocol distribution...")
    
    if "PROTOCOL" in df.columns:
        protocol_dist = df.group_by("PROTOCOL").agg(pl.count().alias("count")).sort("count", descending=True)
        
        # Common protocol mapping
        protocol_map = {6: "TCP", 17: "UDP", 1: "ICMP", 103: "Other"}
        
        results["protocol_distribution"] = {
            "raw_counts": {
                int(row["PROTOCOL"]): int(row["count"])
                for row in protocol_dist.iter_rows(named=True)
            },
            "protocol_names": {
                protocol_map.get(int(row["PROTOCOL"]), f"Protocol_{row['PROTOCOL']}"): {
                    "count": int(row["count"]),
                    "percentage": round((row["count"] / len(df)) * 100, 4)
                }
                for row in protocol_dist.iter_rows(named=True)
            }
        }
    
    # L7 Protocol analysis
    if "L7_PROTO" in df.columns:
        l7_dist = df.group_by("L7_PROTO").agg(pl.count().alias("count")).sort("count", descending=True).head(20)
        
        results["layer7_protocol_distribution"] = {
            "top_20_protocols": {
                int(row["L7_PROTO"]): int(row["count"])
                for row in l7_dist.iter_rows(named=True)
            }
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing statistics for numerical features...")
    
    # Identify numerical columns (exclude IP addresses and labels)
    exclude_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "Label"]
    numerical_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in numeric_types]
    
    numerical_stats = {}
    
    for col in numerical_cols:
        try:
            stats = df.select([
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).median().alias("median"),
                pl.col(col).std().alias("std"),
                pl.col(col).quantile(0.25).alias("q25"),
                pl.col(col).quantile(0.75).alias("q75"),
            ]).to_dicts()[0]
            
            # Convert to native Python types for JSON serialization
            numerical_stats[col] = {
                k: float(v) if v is not None and str(v) != "nan" else 0.0
                for k, v in stats.items()
            }
            
            # Add zero count
            zero_count = df.filter(pl.col(col) == 0).height
            numerical_stats[col]["zero_count"] = zero_count
            numerical_stats[col]["zero_percentage"] = round((zero_count / len(df)) * 100, 4)
            
        except Exception as e:
            print(f"Warning: Could not compute stats for {col}: {e}")
    
    results["numerical_statistics"] = numerical_stats
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing flow duration patterns...")
    
    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        duration_bins = df.select([
            (pl.col("FLOW_DURATION_MILLISECONDS") == 0).sum().alias("zero_duration"),
            ((pl.col("FLOW_DURATION_MILLISECONDS") > 0) & (pl.col("FLOW_DURATION_MILLISECONDS") <= 100)).sum().alias("0-100ms"),
            ((pl.col("FLOW_DURATION_MILLISECONDS") > 100) & (pl.col("FLOW_DURATION_MILLISECONDS") <= 1000)).sum().alias("100ms-1s"),
            ((pl.col("FLOW_DURATION_MILLISECONDS") > 1000) & (pl.col("FLOW_DURATION_MILLISECONDS") <= 10000)).sum().alias("1s-10s"),
            ((pl.col("FLOW_DURATION_MILLISECONDS") > 10000) & (pl.col("FLOW_DURATION_MILLISECONDS") <= 60000)).sum().alias("10s-1min"),
            (pl.col("FLOW_DURATION_MILLISECONDS") > 60000).sum().alias("over_1min"),
        ]).to_dicts()[0]
        
        results["flow_duration_distribution"] = {
            k: {"count": int(v), "percentage": round((v / len(df)) * 100, 4)}
            for k, v in duration_bins.items()
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing traffic volume patterns...")
    
    if "IN_BYTES" in df.columns and "OUT_BYTES" in df.columns:
        # Total traffic statistics
        total_in_bytes = df["IN_BYTES"].sum()
        total_out_bytes = df["OUT_BYTES"].sum()
        
        results["traffic_volume"] = {
            "total_inbound_bytes": int(total_in_bytes),
            "total_outbound_bytes": int(total_out_bytes),
            "total_traffic_bytes": int(total_in_bytes + total_out_bytes),
            "total_inbound_gb": round(total_in_bytes / (1024**3), 4),
            "total_outbound_gb": round(total_out_bytes / (1024**3), 4),
            "total_traffic_gb": round((total_in_bytes + total_out_bytes) / (1024**3), 4),
            "avg_inbound_bytes_per_flow": round(float(df["IN_BYTES"].mean()), 2),
            "avg_outbound_bytes_per_flow": round(float(df["OUT_BYTES"].mean()), 2),
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing packet count patterns...")
    
    if "IN_PKTS" in df.columns and "OUT_PKTS" in df.columns:
        total_in_pkts = df["IN_PKTS"].sum()
        total_out_pkts = df["OUT_PKTS"].sum()
        
        results["packet_statistics"] = {
            "total_inbound_packets": int(total_in_pkts),
            "total_outbound_packets": int(total_out_pkts),
            "total_packets": int(total_in_pkts + total_out_pkts),
            "avg_inbound_packets_per_flow": round(float(df["IN_PKTS"].mean()), 2),
            "avg_outbound_packets_per_flow": round(float(df["OUT_PKTS"].mean()), 2),
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing port usage...")
    
    if "L4_DST_PORT" in df.columns:
        # Top destination ports
        top_dst_ports = df.group_by("L4_DST_PORT").agg(pl.count().alias("count")).sort("count", descending=True).head(30)
        
        # Common port mapping
        common_ports = {
            80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP", 25: "SMTP",
            53: "DNS", 3389: "RDP", 3306: "MySQL", 5432: "PostgreSQL",
            8080: "HTTP-Alt", 1900: "SSDP", 123: "NTP", 547: "DHCPv6"
        }
        
        results["destination_port_analysis"] = {
            "top_30_ports": [
                {
                    "port": int(row["L4_DST_PORT"]),
                    "service": common_ports.get(int(row["L4_DST_PORT"]), "Unknown"),
                    "count": int(row["count"]),
                    "percentage": round((row["count"] / len(df)) * 100, 4)
                }
                for row in top_dst_ports.iter_rows(named=True)
            ]
        }
    
    if "L4_SRC_PORT" in df.columns:
        # Source port statistics (check for ephemeral vs well-known)
        src_port_analysis = df.select([
            (pl.col("L4_SRC_PORT") < 1024).sum().alias("well_known_ports"),
            ((pl.col("L4_SRC_PORT") >= 1024) & (pl.col("L4_SRC_PORT") < 49152)).sum().alias("registered_ports"),
            (pl.col("L4_SRC_PORT") >= 49152).sum().alias("ephemeral_ports"),
        ]).to_dicts()[0]
        
        results["source_port_distribution"] = {
            k: {"count": int(v), "percentage": round((v / len(df)) * 100, 4)}
            for k, v in src_port_analysis.items()
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing TCP flags...")
    
    if "TCP_FLAGS" in df.columns:
        tcp_flag_dist = df.group_by("TCP_FLAGS").agg(pl.count().alias("count")).sort("count", descending=True).head(20)
        
        results["tcp_flags_distribution"] = {
            "top_20_flag_combinations": [
                {
                    "flags": int(row["TCP_FLAGS"]),
                    "count": int(row["count"]),
                    "percentage": round((row["count"] / len(df)) * 100, 4)
                }
                for row in tcp_flag_dist.iter_rows(named=True)
            ]
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing packet size distribution...")
    
    size_cols = [
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES"
    ]
    
    if all(col in df.columns for col in size_cols):
        size_totals = {}
        for col in size_cols:
            size_totals[col] = int(df[col].sum())
        
        total_sized_packets = sum(size_totals.values())
        
        results["packet_size_distribution"] = {
            "total_packets_by_size": size_totals,
            "size_percentages": {
                k: round((v / total_sized_packets * 100) if total_sized_packets > 0 else 0, 4)
                for k, v in size_totals.items()
            }
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing retransmissions...")
    
    retrans_cols = ["RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", 
                    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"]
    
    if all(col in df.columns for col in retrans_cols):
        flows_with_retrans = df.filter(
            (pl.col("RETRANSMITTED_IN_PKTS") > 0) | (pl.col("RETRANSMITTED_OUT_PKTS") > 0)
        ).height
        
        results["retransmission_analysis"] = {
            "flows_with_retransmissions": flows_with_retrans,
            "retransmission_percentage": round((flows_with_retrans / len(df)) * 100, 4),
            "total_retransmitted_in_bytes": int(df["RETRANSMITTED_IN_BYTES"].sum()),
            "total_retransmitted_out_bytes": int(df["RETRANSMITTED_OUT_BYTES"].sum()),
            "total_retransmitted_in_packets": int(df["RETRANSMITTED_IN_PKTS"].sum()),
            "total_retransmitted_out_packets": int(df["RETRANSMITTED_OUT_PKTS"].sum()),
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing temporal patterns...")
    
    if "FLOW_START_MILLISECONDS" in df.columns and "FLOW_END_MILLISECONDS" in df.columns:
        # Convert milliseconds to datetime
        df_time = df.with_columns([
            (pl.col("FLOW_START_MILLISECONDS") / 1000).cast(pl.Int64).alias("start_seconds"),
            (pl.col("FLOW_END_MILLISECONDS") / 1000).cast(pl.Int64).alias("end_seconds")
        ])
        
        start_min = df_time["start_seconds"].min()
        start_max = df_time["start_seconds"].max()
        end_min = df_time["end_seconds"].min()
        end_max = df_time["end_seconds"].max()
        
        results["temporal_analysis"] = {
            "dataset_start_timestamp": int(start_min) if start_min else 0,
            "dataset_end_timestamp": int(end_max) if end_max else 0,
            "dataset_start_datetime": datetime.fromtimestamp(int(start_min)).isoformat() if start_min else "N/A",
            "dataset_end_datetime": datetime.fromtimestamp(int(end_max)).isoformat() if end_max else "N/A",
            "time_span_seconds": int(end_max - start_min) if start_min and end_max else 0,
            "time_span_hours": round((end_max - start_min) / 3600, 2) if start_min and end_max else 0,
            "time_span_days": round((end_max - start_min) / 86400, 2) if start_min and end_max else 0,
        }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing attack patterns...")
    
    if "Attack" in df.columns and "Label" in df.columns:
        # Compare benign vs attack traffic characteristics
        benign_df = df.filter(pl.col("Attack") == "Benign")
        attack_df = df.filter(pl.col("Attack") != "Benign")
        
        comparison_features = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"]
        
        attack_comparison = {}
        
        for feature in comparison_features:
            if feature in df.columns:
                attack_comparison[feature] = {
                    "benign_mean": round(float(benign_df[feature].mean()), 2) if len(benign_df) > 0 else 0,
                    "attack_mean": round(float(attack_df[feature].mean()), 2) if len(attack_df) > 0 else 0,
                    "benign_median": round(float(benign_df[feature].median()), 2) if len(benign_df) > 0 else 0,
                    "attack_median": round(float(attack_df[feature].median()), 2) if len(attack_df) > 0 else 0,
                    "benign_std": round(float(benign_df[feature].std()), 2) if len(benign_df) > 0 else 0,
                    "attack_std": round(float(attack_df[feature].std()), 2) if len(attack_df) > 0 else 0,
                }
        
        results["attack_vs_benign_comparison"] = attack_comparison
        
        # Label type specific statistics
        if len(attack_df) > 0:
            attack_types = attack_df.group_by("Attack").agg([
                pl.count().alias("count"),
                pl.col("IN_BYTES").mean().alias("avg_in_bytes"),
                pl.col("OUT_BYTES").mean().alias("avg_out_bytes"),
                pl.col("FLOW_DURATION_MILLISECONDS").mean().alias("avg_duration_ms"),
            ]).sort("count", descending=True)
            
            results["attack_type_characteristics"] = [
                {
                    "attack_type": row["Attack"],
                    "count": int(row["count"]),
                    "avg_inbound_bytes": round(float(row["avg_in_bytes"]), 2),
                    "avg_outbound_bytes": round(float(row["avg_out_bytes"]), 2),
                    "avg_duration_ms": round(float(row["avg_duration_ms"]), 2),
                }
                for row in attack_types.iter_rows(named=True)
            ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing inter-arrival time patterns...")
    
    iat_features = [
        "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"
    ]
    
    iat_summary = {}
    for feature in iat_features:
        if feature in df.columns:
            # Filter out zero values for meaningful statistics
            non_zero = df.filter(pl.col(feature) > 0)
            if len(non_zero) > 0:
                iat_summary[feature] = {
                    "non_zero_count": len(non_zero),
                    "mean": round(float(non_zero[feature].mean()), 4),
                    "median": round(float(non_zero[feature].median()), 4),
                    "min": round(float(non_zero[feature].min()), 4),
                    "max": round(float(non_zero[feature].max()), 4),
                }
    
    results["inter_arrival_time_analysis"] = iat_summary
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing feature cardinality...")
    
    cardinality = {}
    for col in df.columns:
        try:
            unique_count = df[col].n_unique()
            cardinality[col] = {
                "unique_values": unique_count,
                "cardinality_ratio": round(unique_count / len(df), 6)
            }
        except:
            pass
    
    results["feature_cardinality"] = cardinality
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing data quality metrics...")
    
    # Check for duplicate rows
    duplicate_count = len(df) - df.unique().height
    
    # Check for flows with zero bytes and packets
    zero_traffic = df.filter(
        (pl.col("IN_BYTES") == 0) & (pl.col("OUT_BYTES") == 0) &
        (pl.col("IN_PKTS") == 0) & (pl.col("OUT_PKTS") == 0)
    ).height if all(c in df.columns for c in ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]) else 0
    
    results["data_quality"] = {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": round((duplicate_count / len(df)) * 100, 4),
        "zero_traffic_flows": zero_traffic,
        "zero_traffic_percentage": round((zero_traffic / len(df)) * 100, 4),
        "completeness_score": round(100 - (sum(v["null_count"] for v in missing_analysis.values()) / (len(df) * len(df.columns)) * 100), 2)
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing feature relationships...")
    
    # Check correlation between related features
    correlation_insights = {}
    
    if "IN_BYTES" in df.columns and "IN_PKTS" in df.columns:
        # Average packet size
        avg_pkt_size = df.filter(pl.col("IN_PKTS") > 0).select(
            (pl.col("IN_BYTES") / pl.col("IN_PKTS")).alias("avg_pkt_size")
        )
        if len(avg_pkt_size) > 0:
            correlation_insights["inbound_avg_packet_size"] = {
                "mean": round(float(avg_pkt_size["avg_pkt_size"].mean()), 2),
                "median": round(float(avg_pkt_size["avg_pkt_size"].median()), 2),
            }
    
    if "OUT_BYTES" in df.columns and "OUT_PKTS" in df.columns:
        avg_pkt_size = df.filter(pl.col("OUT_PKTS") > 0).select(
            (pl.col("OUT_BYTES") / pl.col("OUT_PKTS")).alias("avg_pkt_size")
        )
        if len(avg_pkt_size) > 0:
            correlation_insights["outbound_avg_packet_size"] = {
                "mean": round(float(avg_pkt_size["avg_pkt_size"].mean()), 2),
                "median": round(float(avg_pkt_size["avg_pkt_size"].median()), 2),
            }
    
    results["correlation_insights"] = correlation_insights
    
    # Calculate class imbalance ratio
    if "Label" in df.columns:
        attack_counts = df.group_by("Label").agg(pl.count().alias("count"))
        counts_dict = {row["Label"]: row["count"] for row in attack_counts.iter_rows(named=True)}
        
        if 0 in counts_dict and 1 in counts_dict:
            imbalance_ratio = counts_dict[0] / counts_dict[1] if counts_dict[1] > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
    else:
        imbalance_ratio = 1.0
    
    results["summary"] = {
        "total_flows_analyzed": len(df),
        "total_features": len(df.columns),
        "benign_flows": int(counts_dict.get(0, 0)) if "Label" in df.columns else 0,
        "attack_flows": int(counts_dict.get(1, 0)) if "Label" in df.columns else 0,
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "unique_attack_types": results.get("attack_distribution", {}).get("unique_labels", 0) - 1 if "Attack" in df.columns else 0,
        "data_quality_score": results["data_quality"]["completeness_score"],
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis complete!")
    
    return results
