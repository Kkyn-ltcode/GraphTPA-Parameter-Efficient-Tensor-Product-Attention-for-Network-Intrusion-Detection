import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from typing import Optional, Tuple


class TPAConv(MessagePassing):
    """
    Tensor Product Attention Convolution for Graph Neural Networks.
    
    Implements TPA from "Tensor Product Attention Is All You Need" paper,
    adapted for graph convolution operations with RoPE support.
    
    Args:
        in_channels (int): Size of input features
        out_channels (int): Size of output features per head
        heads (int): Number of attention heads
        rank_q (int): Rank for Query factorization
        rank_k (int): Rank for Key factorization  
        rank_v (int): Rank for Value factorization
        dropout (float): Dropout probability
        edge_dim (int, optional): Edge feature dimensionality
        bias (bool): Whether to use bias
        beta (bool): Whether to use gating mechanism
        use_rope (bool): Whether to use Rotary Position Embedding
        rope_theta (float): Base theta for RoPE
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        rank_q: int = None,
        rank_k: int = None,
        rank_v: int = None,
        dropout: float = 0.0,
        edge_dim: int = None,
        bias: bool = True,
        beta: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.beta = beta
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        
        # Set default ranks to in_channels // 4 if not specified
        self.rank_q = rank_q if rank_q is not None else max(in_channels // 4, 32)
        self.rank_k = rank_k if rank_k is not None else max(in_channels // 4, 32)
        self.rank_v = rank_v if rank_v is not None else max(in_channels // 4, 32)
        
        # Cache for storing low-rank KV representations
        self.kv_cache = None
        self.cache_enabled = False
        
        # Query factorization: Q = (1/R_Q) * A_Q^T @ B_Q
        # A_Q: [in_channels, rank_q], B_Q: [rank_q, heads * out_channels]
        self.lin_A_q = Linear(in_channels, self.rank_q, bias=False, weight_initializer='glorot')
        self.lin_B_q = Linear(self.rank_q, heads * out_channels, bias=bias, weight_initializer='glorot')
        
        # Key factorization: K = (1/R_K) * A_K^T @ B_K
        self.lin_A_k = Linear(in_channels, self.rank_k, bias=False, weight_initializer='glorot')
        self.lin_B_k = Linear(self.rank_k, heads * out_channels, bias=bias, weight_initializer='glorot')
        
        # Value factorization: V = (1/R_V) * A_V^T @ B_V
        self.lin_A_v = Linear(in_channels, self.rank_v, bias=False, weight_initializer='glorot')
        self.lin_B_v = Linear(self.rank_v, heads * out_channels, bias=bias, weight_initializer='glorot')
        
        # Edge feature projection (if provided)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.lin_edge = None
            
        # Gating mechanism (from TransformerConv)
        if beta:
            self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False, weight_initializer='glorot')
        else:
            self.lin_beta = None
        
        # RoPE: Precompute frequency bands
        if use_rope:
            self._init_rope_freqs()
        else:
            self.register_buffer('rope_freqs', None)
            
        self.reset_parameters()
    
    def _init_rope_freqs(self):
        """Initialize RoPE frequency bands"""
        # For rank dimensions, create frequency bands
        # freq_i = theta^(-2i/d) for i in [0, d/2)
        d = max(self.rank_q, self.rank_k)
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('rope_freqs', freqs)
    
    def _apply_rope(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply Rotary Position Embedding to tensor.
        
        Args:
            x: Input tensor [batch, rank_dim]
            positions: Position indices [batch] (optional, defaults to sequential)
            
        Returns:
            x_rotated: Tensor with RoPE applied [batch, rank_dim]
        """
        if not self.use_rope or self.rope_freqs is None:
            return x
        
        batch_size, dim = x.shape
        
        # Default to sequential positions if not provided
        if positions is None:
            positions = torch.arange(batch_size, device=x.device, dtype=x.dtype)
        
        # Expand positions for frequency bands
        positions = positions.unsqueeze(-1)  # [batch, 1]
        freqs = self.rope_freqs[:dim // 2]  # [dim/2]
        
        # Compute angles: pos * freq
        angles = positions * freqs  # [batch, dim/2]
        
        # Compute cos and sin
        cos_pos = torch.cos(angles)  # [batch, dim/2]
        sin_pos = torch.sin(angles)  # [batch, dim/2]
        
        # Split x into pairs for rotation
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation: [cos -sin] [x1]
        #                 [sin  cos] [x2]
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x1 * cos_pos - x2 * sin_pos
        x_rotated[..., 1::2] = x1 * sin_pos + x2 * cos_pos
        
        return x_rotated
        
    def reset_parameters(self):
        """Initialize parameters"""
        self.lin_A_q.reset_parameters()
        self.lin_B_q.reset_parameters()
        self.lin_A_k.reset_parameters()
        self.lin_B_k.reset_parameters()
        self.lin_A_v.reset_parameters()
        self.lin_B_v.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()
            
    def enable_kv_cache(self):
        """Enable KV cache for inference efficiency"""
        self.cache_enabled = True
        self.kv_cache = {}
        
    def disable_kv_cache(self):
        """Disable and clear KV cache"""
        self.cache_enabled = False
        self.kv_cache = None
        
    def get_nuclear_norm_loss(self) -> Tensor:
        """
        Compute nuclear norm regularization loss for low-rank factors.
        Encourages low-rank structure in the factorization.
        
        Returns:
            nuclear_loss: Sum of nuclear norms of factor products
        """
        device = self.lin_A_q.weight.device
        nuclear_loss = torch.tensor(0.0, device=device)
        
        # For each Q, K, V compute nuclear norm of A @ B (approximate)
        # Using Frobenius norm as approximation for efficiency
        # ||A @ B||_* ≈ ||A||_F * ||B||_F
        nuclear_loss += torch.norm(self.lin_A_q.weight, p='fro') * torch.norm(self.lin_B_q.weight, p='fro')
        nuclear_loss += torch.norm(self.lin_A_k.weight, p='fro') * torch.norm(self.lin_B_k.weight, p='fro')
        nuclear_loss += torch.norm(self.lin_A_v.weight, p='fro') * torch.norm(self.lin_B_v.weight, p='fro')
        
        return nuclear_loss / 3.0  # Average across Q, K, V
    
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        positions: OptTensor = None,
        return_attention_weights: bool = False
    ):
        """
        Forward pass of TPA convolution.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            positions: Position indices for RoPE [num_nodes] (optional)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            out: Output node features [num_nodes, heads * out_channels]
            attention_weights (optional): Attention weights
        """
        H, C = self.heads, self.out_channels
        
        # Compute Query using tensor product factorization
        # Q = (1/R_Q) * A_Q(x)^T @ B_Q(A_Q(x))
        A_q = self.lin_A_q(x)  # [num_nodes, rank_q]
        B_q = self.lin_B_q(A_q)  # [num_nodes, H * C]
        
        # Apply RoPE to B_Q (as per TPA paper Figure 1)
        if self.use_rope:
            B_q = self._apply_rope(B_q, positions)
        
        query = B_q / math.sqrt(self.rank_q)  # [num_nodes, H * C]
        
        # Compute Key using tensor product factorization with caching support
        if self.cache_enabled and self.kv_cache is not None and 'A_k' in self.kv_cache:
            # Use cached low-rank factors
            A_k = self.kv_cache['A_k']
            B_k = self.kv_cache['B_k']
            A_v = self.kv_cache['A_v']
            B_v = self.kv_cache['B_v']
        else:
            A_k = self.lin_A_k(x)  # [num_nodes, rank_k]
            B_k = self.lin_B_k(A_k)  # [num_nodes, H * C]
            
            # Apply RoPE to B_K (as per TPA paper Figure 1)
            if self.use_rope:
                B_k = self._apply_rope(B_k, positions)
            
            A_v = self.lin_A_v(x)  # [num_nodes, rank_v]
            B_v = self.lin_B_v(A_v)  # [num_nodes, H * C]
            
            # Cache the low-rank factors if enabled
            if self.cache_enabled:
                self.kv_cache = {
                    'A_k': A_k.detach(),
                    'B_k': B_k.detach(),
                    'A_v': A_v.detach(),
                    'B_v': B_v.detach()
                }
        
        key = B_k / math.sqrt(self.rank_k)  # [num_nodes, H * C]
        value = B_v / math.sqrt(self.rank_v)  # [num_nodes, H * C]
        
        # Reshape for multi-head attention
        query = query.view(-1, H, C)  # [num_nodes, H, C]
        key = key.view(-1, H, C)  # [num_nodes, H, C]
        value = value.view(-1, H, C)  # [num_nodes, H, C]
        
        # Propagate messages
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None
        )
        
        # Gating mechanism (if enabled)
        if self.lin_beta is not None:
            # Flatten for beta computation: [num_nodes, H, C] -> [num_nodes, H * C]
            query_flat = query.view(-1, H * C)
            out_flat = out.view(-1, H * C)
            beta = self.lin_beta(torch.cat([query_flat, out_flat, query_flat - out_flat], dim=-1))
            beta = beta.sigmoid()
            # Apply gating to the multi-head tensors
            out = beta.view(-1, 1, 1) * out + (1 - beta.view(-1, 1, 1)) * query
            
        out = out.view(-1, H * C)  # [num_nodes, H * C]
        
        if return_attention_weights:
            # Note: returning attention weights requires storing them in message()
            # For now, return None as placeholder
            return out, None
        else:
            return out
            
    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: int
    ) -> Tensor:
        """
        Compute messages for each edge.
        
        Args:
            query_i: Query from target nodes [num_edges, H, C]
            key_j: Key from source nodes [num_edges, H, C]
            value_j: Value from source nodes [num_edges, H, C]
            edge_attr: Edge attributes [num_edges, edge_dim]
            index: Target node indices [num_edges]
            ptr: CSR pointer (optional)
            size_i: Number of target nodes
            
        Returns:
            msg: Messages [num_edges, H, C]
        """
        # Compute attention scores: Q @ K^T / sqrt(d)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)  # [num_edges, H]
        
        # Add edge features to attention scores (if provided)
        if edge_attr is not None and self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (query_i * edge_emb).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # Softmax normalization over neighbors
        alpha = softmax(alpha, index, ptr, size_i)  # [num_edges, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight values by attention scores
        out = value_j * alpha.unsqueeze(-1)  # [num_edges, H, C]
        
        return out
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'rank_q={self.rank_q}, rank_k={self.rank_k}, '
                f'rank_v={self.rank_v})')
