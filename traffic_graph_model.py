import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, TransformerConv, GCNConv, SAGEConv
from tpa_conv import TPAConv
from tensor_edge_repr import TensorEdgeRepresentation, HybridEdgeRepresentation

class GraphTransformer(nn.Module):
    """
    Simplified Graph Transformer for edge classification.
    
    Key changes to reduce overfitting:
    - Removed BatchNorm (keeps only LayerNorm for stable train/eval behavior)
    - GCN and SAGE are disabled by default (use only TransformerConv or TPAConv)
    - Added more dropout points
    - Simplified fusion layer
    - Optional TPA (Tensor Product Attention) for memory efficiency
    """
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, 
                 num_layers=3, num_heads=4, dropout=0.3, use_edge_features=True, 
                 useTrans=True, useGCN=True, useSAGE=True, use_tpa=True, use_rope=False, 
                 rank_adaptation='hierarchical', use_tensor_edge=True, 
                 tensor_edge_rank=None, tensor_edge_mode='tensor', activation='gelu'):  
        """
        Args:
            use_tensor_edge: Whether to use tensor product for edge representation
            tensor_edge_rank: Rank for tensor edge factorization (None = auto)
            tensor_edge_mode: 'tensor', 'hybrid', or 'concat' (standard)
        """
        super().__init__()
        
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_edge_features = use_edge_features
        self.useTrans = useTrans
        self.useGCN = useGCN
        self.useSAGE = useSAGE
        self.use_tpa = use_tpa
        self.use_rope = use_rope
        self.rank_adaptation = rank_adaptation
        self.use_tensor_edge = use_tensor_edge
        self.tensor_edge_rank = tensor_edge_rank
        self.tensor_edge_mode = tensor_edge_mode

        self.num_blocks = 0
        if self.use_tpa:
            self.num_blocks += 1
        if self.useTrans:
            self.num_blocks += 1
        if self.useGCN:
            self.num_blocks += 1
        if self.useSAGE:
            self.num_blocks += 1

        if activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'silu':
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.GELU()
        
        # Compute adaptive ranks per layer
        self.layer_ranks = self._compute_layer_ranks()
        
        # Initialize all layers
        self._init_layers()

    def _compute_layer_ranks(self):
        """
        Compute adaptive ranks for each layer based on rank_adaptation strategy.
        
        Strategies:
        - 'hierarchical': Higher ranks in early layers, lower in later layers
        - 'uniform': Same rank across all layers
        - 'custom': Can be extended for other patterns
        """
        ranks = []
        base_rank = max(self.hidden_channels // 4, 32)
        
        if self.rank_adaptation == 'hierarchical':
            # Early layers: high rank (fine details)
            # Later layers: low rank (abstract patterns)
            for layer_idx in range(self.num_layers):
                # Exponential decay: rank_i = base_rank * 2^((num_layers - i - 1) / num_layers)
                scale = 2.0 ** ((self.num_layers - layer_idx - 1) / self.num_layers)
                rank = max(int(base_rank * scale), 16)
                ranks.append(rank)
        elif self.rank_adaptation == 'uniform':
            # Same rank for all layers
            ranks = [base_rank] * self.num_layers
        else:
            # Default to uniform if unknown strategy
            ranks = [base_rank] * self.num_layers
        
        return ranks
    
    def _init_layers(self):
        """Initialize all layers (separated from __init__ for clarity)"""
        # Input projections with dropout
        self.node_input_proj = nn.Sequential(
            nn.Linear(self.node_in_channels, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.act_fn,
            nn.Dropout(self.dropout),
        )

        if self.use_edge_features:
            self.edge_input_proj = nn.Sequential(
                nn.Linear(self.edge_in_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                self.act_fn,
                nn.Dropout(self.dropout),
            )
        
        self.tpa_convs = nn.ModuleList()
        self.trans_convs = nn.ModuleList()
        self.gcn_convs = nn.ModuleList()
        self.sage_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Only LayerNorm, no BatchNorm
        self.pre_norm = nn.ModuleList()  # Pre-normalization for better training
            
        for layer_idx in range(self.num_layers):
            # Pre-normalization layer
            self.pre_norm.append(nn.LayerNorm(self.hidden_channels))
            
            if self.use_tpa:
                # Use Tensor Product Attention convolution with adaptive ranks
                rank = self.layer_ranks[layer_idx]
                conv = TPAConv(
                    self.hidden_channels,
                    self.hidden_channels // self.num_heads,
                    heads=self.num_heads,
                    rank_q=rank,
                    rank_k=rank,
                    rank_v=rank,
                    dropout=self.dropout,
                    edge_dim=self.hidden_channels if self.use_edge_features else None,
                    beta=True,
                    use_rope=self.use_rope,
                    rope_theta=10000.0
                )
                self.tpa_convs.append(conv)

            if self.useTrans:
                # Use standard TransformerConv
                conv = TransformerConv(
                    self.hidden_channels, 
                    self.hidden_channels // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout,
                    beta=True,
                    edge_dim=self.hidden_channels if self.use_edge_features else None
                )
                self.trans_convs.append(conv)

            if self.useGCN:
                self.gcn_convs.append(
                    GCNConv(
                        in_channels=self.hidden_channels, 
                        out_channels=self.hidden_channels,
                        add_self_loops=True,
                        normalize=True
                    )
                )

            if self.useSAGE:
                self.sage_convs.append(
                    SAGEConv(
                        in_channels=self.hidden_channels, 
                        out_channels=self.hidden_channels,
                        aggr='mean',
                        normalize=True
                    )
                )

            # Post-layer normalization (only LayerNorm, removed BatchNorm)
            # self.layer_norms.append(nn.LayerNorm(self.hidden_channels))
            self.layer_norms.append(nn.LayerNorm(self.hidden_channels * self.num_blocks))

        # Simplified fusion layer
        if self.num_blocks > 1:
            self.fusion = nn.Sequential(
                nn.Linear(self.num_blocks * self.hidden_channels, self.hidden_channels),
                # nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                self.act_fn,
                nn.Dropout(self.dropout),
            )
        else:
            self.fusion = nn.Identity()

        # Initialize tensor edge representation module
        if self.use_tensor_edge and self.tensor_edge_mode != 'concat':
            if self.tensor_edge_mode == 'hybrid':
                self.edge_repr_module = HybridEdgeRepresentation(
                    hidden_channels=self.hidden_channels,
                    rank=self.tensor_edge_rank,
                    use_edge_features=self.use_edge_features,
                    edge_feature_channels=self.hidden_channels if self.use_edge_features else None,
                    dropout=self.dropout,
                    alpha=0.5
                )
            else:  # 'tensor' mode
                self.edge_repr_module = TensorEdgeRepresentation(
                    hidden_channels=self.hidden_channels,
                    rank=self.tensor_edge_rank,
                    use_edge_features=self.use_edge_features,
                    edge_feature_channels=self.hidden_channels if self.use_edge_features else None,
                    dropout=self.dropout
                )
            # Tensor edge module outputs hidden_channels directly
            edge_repr_dim = self.hidden_channels
        else:
            # Standard concatenation approach
            self.edge_repr_module = None
            edge_repr_dim = self.hidden_channels * 2
            if self.use_edge_features:
                edge_repr_dim += self.hidden_channels

        # Simplified output projection with more dropout
        self.output_proj = nn.Sequential(
            Linear(edge_repr_dim, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            self.act_fn,
            nn.Dropout(self.dropout),
            Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.LayerNorm(self.hidden_channels // 2),
            self.act_fn,
            nn.Dropout(self.dropout * 1.5),  # Higher dropout before final layer
            Linear(self.hidden_channels // 2, self.out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None, edge_label_attr=None, edge_label_index=None, positions=None):
        x = self.node_input_proj(x)

        if self.use_edge_features and edge_attr is not None:
            edge_attr_proj = self.edge_input_proj(edge_attr)
        else:
            edge_attr_proj = None

        for i in range(self.num_layers):
            x_residual = x
            
            # Pre-normalization (more stable training)
            x_normed = self.pre_norm[i](x)
            
            gnn_outputs = []

            if self.use_tpa:
                x_tpa = self.tpa_convs[i](x=x_normed, edge_index=edge_index, edge_attr=edge_attr_proj)
                gnn_outputs.append(x_tpa)

            if self.useTrans:
                x_trans = self.trans_convs[i](x=x_normed, edge_index=edge_index, edge_attr=edge_attr_proj)
                gnn_outputs.append(x_trans)

            if self.useGCN:
                x_gcn = self.gcn_convs[i](x=x_normed, edge_index=edge_index)
                gnn_outputs.append(x_gcn)

            if self.useSAGE:
                x_sage = self.sage_convs[i](x=x_normed, edge_index=edge_index)
                gnn_outputs.append(x_sage)

            # Concatenate and normalize (only LayerNorm, no BatchNorm)
            if len(gnn_outputs) > 1:
                x_multi = torch.cat(gnn_outputs, dim=-1)
                # x_multi = torch.stack(gnn_outputs, dim=0).sum(dim=0)
                x_multi = self.layer_norms[i](x_multi)
                x_fused = self.fusion(x_multi)
            else:
                x_fused = gnn_outputs[0]
                x_fused = self.layer_norms[i](x_fused)
            
            # Residual connection with dropout
            x = x_residual + F.dropout(x_fused, p=self.dropout, training=self.training)

            if i < self.num_layers - 1:
                x = F.gelu(x)

        if edge_label_index is None:
            edge_label_index = edge_index

        src_emb = x[edge_label_index[0]]
        dst_emb = x[edge_label_index[1]]

        # Compute edge representation
        if self.edge_repr_module is not None:
            # Use tensor product-based edge representation
            edge_feat = None
            if self.use_edge_features and edge_label_attr is not None:
                edge_feat = self.edge_input_proj(edge_label_attr)
            edge_emb = self.edge_repr_module(src_emb, dst_emb, edge_feat)
        else:
            # Standard concatenation approach
            edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
            if self.use_edge_features and edge_label_attr is not None:
                edge_feat = self.edge_input_proj(edge_label_attr)
                edge_emb = torch.cat([edge_emb, edge_feat], dim=-1)

        return edge_emb, self.output_proj(edge_emb)
