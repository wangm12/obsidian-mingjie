import torch
import torch.nn as nn
import math
from einops import einsum
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        # Initialize the parent class
        super().__init__()
        
        # params
        self.d_in = in_features
        self.d_out = out_features
        self.device = device
        self.dtype = dtype
        
        # Initialize weight
        self.weight: Float[torch.Tensor, "d_out, d_in"] = self._init_weight()
        

    def _init_weight(self):
        std = math.sqrt(2 / (self.d_in + self.d_out))
        # We need the weight to be stored as W (d_out, d_in)
        empty_weight = torch.empty(self.d_out, self.d_in, device=self.device, dtype=self.dtype)
        return nn.Parameter(
            nn.init.trunc_normal_(empty_weight, mean=0.0, std=std, a=-3*std, b=3*std),
            # lets log all changes to the parameter
            requires_grad=True
        )
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> torch.Tensor:
        # linear transformation
        return einsum(
            x, self.weight,
            "... d_in, d_out d_in -> ... d_out",
        )


class Embedding(nn.Module):
    def __init__(self, 
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Initialize the parent class
        super().__init__()
        
        # params
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        # Initialize weight
        self.weight: Float[torch.Tensor, "num_embeddings, embedding_dim"] = self._init_weight()
        
    def _init_weight(self):
        empty_weight = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype)
        return nn.Parameter(
            nn.init.trunc_normal_(empty_weight, mean=0.0, std=1.0, a=-3.0, b=3.0),
            # lets log all changes to the parameter
            requires_grad=True
        )
    
    def forward(self, token_ids: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        """ 
        token_ids: (B, S) 
        return: (B, S, D) where D is the embedding dimension
        
        假设 weight 形状为 [1000, 512]，vocab_size=1000, d_model=512
        token_ids = [5, 10, 15]  # 三个token的ID
        embeddings = weight[token_ids]  # 获取第5、10、15行
        结果形状: [3, 512]
        """
        return self.weight[token_ids]