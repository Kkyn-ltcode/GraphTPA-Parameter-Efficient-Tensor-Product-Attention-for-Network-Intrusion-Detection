import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorEdgeRepresentation(nn.Module):
    """
    Tensor Product-based edge representation for graph neural networks.
    
    Instead of: edge_emb = concat(src_emb, dst_emb, edge_feat)
    We use: edge_emb = TensorProduct(A_src, B_src, A_dst, B_dst) + edge_feat
    
    This captures richer interactions between source and destination nodes while
    using fewer parameters through low-rank factorization.
    """
    
    def __init__(
        self, 
        hidden_channels: int,
        rank: int = None,
        use_edge_features: bool = True,
        edge_feature_channels: int = None,
        dropout: float = 0.3,
        activation: str = 'gelu'
    ):
        """
        Args:
            hidden_channels: Dimensionality of node embeddings
            rank: Rank for tensor factorization. If None, defaults to hidden_channels // 4
            use_edge_features: Whether to incorporate edge features
            edge_feature_channels: Dimension of edge features (if used)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.rank = rank if rank is not None else max(hidden_channels // 4, 32)
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        
        # Low-rank factor projections for source nodes
        self.A_src_proj = nn.Linear(hidden_channels, self.rank, bias=False)
        self.B_src_proj = nn.Linear(hidden_channels, self.rank, bias=False)
        
        # Low-rank factor projections for destination nodes
        self.A_dst_proj = nn.Linear(hidden_channels, self.rank, bias=False)
        self.B_dst_proj = nn.Linear(hidden_channels, self.rank, bias=False)
        
        # Compute output dimension after tensor product
        # We'll create: A_src ⊗ B_src + A_dst ⊗ B_dst (element-wise operations)
        tensor_product_dim = self.rank * 2  # Concatenate src and dst tensor products
        
        # If using edge features, add them
        if self.use_edge_features and edge_feature_channels is not None:
            self.edge_feat_proj = nn.Linear(edge_feature_channels, self.rank)
            total_dim = tensor_product_dim + self.rank
        else:
            self.edge_feat_proj = None
            total_dim = tensor_product_dim
        
        # Output projection to get back to hidden_channels dimension
        # This is more efficient than concat(src, dst) which would be 2*hidden_channels
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
        
        self.output_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            act_fn,
            nn.Dropout(dropout)
        )
        
        # Optional: Learnable scaling factor (similar to TPA's 1/R scaling)
        self.scale = nn.Parameter(torch.ones(1) / (self.rank ** 0.5))
        
    def forward(
        self, 
        src_emb: torch.Tensor, 
        dst_emb: torch.Tensor, 
        edge_feat: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute tensor product-based edge representation.
        
        Args:
            src_emb: Source node embeddings [num_edges, hidden_channels]
            dst_emb: Destination node embeddings [num_edges, hidden_channels]
            edge_feat: Optional edge features [num_edges, edge_feature_channels]
            
        Returns:
            edge_repr: Edge representation [num_edges, hidden_channels]
        """
        # Project to low-rank factors
        A_src = self.A_src_proj(src_emb)  # [num_edges, rank]
        B_src = self.B_src_proj(src_emb)  # [num_edges, rank]
        
        A_dst = self.A_dst_proj(dst_emb)  # [num_edges, rank]
        B_dst = self.B_dst_proj(dst_emb)  # [num_edges, rank]
        
        # Compute tensor products (element-wise)
        # This captures interactions between A and B factors
        tp_src = A_src * B_src  # [num_edges, rank]
        tp_dst = A_dst * B_dst  # [num_edges, rank]
        
        # Scale the tensor products (similar to TPA's 1/R_Q scaling)
        tp_src = tp_src * self.scale
        tp_dst = tp_dst * self.scale
        
        # Concatenate source and destination tensor products
        edge_repr = torch.cat([tp_src, tp_dst], dim=-1)  # [num_edges, rank*2]
        
        # Incorporate edge features if available
        if self.use_edge_features and edge_feat is not None and self.edge_feat_proj is not None:
            edge_feat_proj = self.edge_feat_proj(edge_feat)  # [num_edges, rank]
            edge_repr = torch.cat([edge_repr, edge_feat_proj], dim=-1)  # [num_edges, rank*2 + rank]
        
        # Project to final hidden dimension
        edge_repr = self.output_proj(edge_repr)  # [num_edges, hidden_channels]
        
        return edge_repr
    
    def get_compression_ratio(self) -> float:
        """
        Calculate parameter compression ratio vs standard concatenation.
        
        Standard approach: concat(src, dst, edge_feat) -> Linear(total_dim, hidden)
        Our approach: Low-rank factorization with tensor products
        
        Returns:
            Compression ratio (higher is better)
        """
        # Standard approach parameters
        standard_input_dim = 2 * self.hidden_channels
        if self.use_edge_features and self.edge_feat_proj is not None:
            standard_input_dim += self.rank  # Approximate edge feature dim
        standard_params = standard_input_dim * self.hidden_channels
        
        # Our approach parameters
        our_params = (
            2 * self.hidden_channels * self.rank +  # A_src, B_src projections
            2 * self.hidden_channels * self.rank +  # A_dst, B_dst projections
            (self.rank * 2 + (self.rank if self.use_edge_features else 0)) * self.hidden_channels  # Output proj
        )
        
        if self.use_edge_features and self.edge_feat_proj is not None:
            our_params += self.rank * self.rank  # Edge feature projection
        
        return standard_params / our_params


class HybridEdgeRepresentation(nn.Module):
    """
    Hybrid approach: Combines tensor product with residual concatenation.
    
    Useful for gradually transitioning from standard approach to tensor product,
    or for cases where you want both explicit concatenation and learned interactions.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        rank: int = None,
        use_edge_features: bool = True,
        edge_feature_channels: int = None,
        dropout: float = 0.3,
        alpha: float = 0.5  # Weight between tensor product and concat
    ):
        super().__init__()
        
        self.alpha = alpha  # Learnable mixing parameter
        self.hidden_channels = hidden_channels
        self.use_edge_features = use_edge_features
        
        # Tensor product branch
        self.tensor_branch = TensorEdgeRepresentation(
            hidden_channels=hidden_channels,
            rank=rank,
            use_edge_features=use_edge_features,
            edge_feature_channels=edge_feature_channels,
            dropout=dropout
        )
        
        # Standard concatenation branch (for residual)
        concat_dim = 2 * hidden_channels
        if use_edge_features and edge_feature_channels is not None:
            concat_dim += edge_feature_channels
        
        self.concat_branch = nn.Sequential(
            nn.Linear(concat_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable mixing weight
        self.mix_weight = nn.Parameter(torch.tensor(alpha))
        
    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_feat: torch.Tensor = None
    ) -> torch.Tensor:
        # Tensor product branch
        tensor_repr = self.tensor_branch(src_emb, dst_emb, edge_feat)
        
        # Concatenation branch
        concat_repr = torch.cat([src_emb, dst_emb], dim=-1)
        if self.use_edge_features and edge_feat is not None:
            concat_repr = torch.cat([concat_repr, edge_feat], dim=-1)
        concat_repr = self.concat_branch(concat_repr)
        
        # Mix both representations
        mix_weight = torch.sigmoid(self.mix_weight)  # Constrain to [0, 1]
        edge_repr = mix_weight * tensor_repr + (1 - mix_weight) * concat_repr
        
        return edge_repr
