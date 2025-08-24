import torch
import torch.nn as nn
import math
from einops import einsum, reduce, rearrange
from jaxtyping import Float, Int

class Linear(nn.Module):
    """
    Linear layer implementation without bias term.
    Performs matrix multiplication: y = xW^T where W has shape (d_out, d_in).
    Uses Xavier/Glorot initialization with truncated normal distribution.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Args:
            in_features: Input dimension (d_in)
            out_features: Output dimension (d_out)
            device: Device to place the weight tensor
            dtype: Data type of the weight tensor
        
        Example:
            in_features=768, out_features=3072
            Creates weight matrix of shape [3072, 768]
            Maps input from 768 dimensions to 3072 dimensions
        """
        # Initialize the parent class
        super().__init__()
        
        # Store dimensions and device info
        self.d_in = in_features
        self.d_out = out_features
        self.device = device
        self.dtype = dtype
        
        # Initialize weight matrix with shape (d_out, d_in)
        self.weight: Float[torch.Tensor, "d_out, d_in"] = self._init_weight()
        

    def _init_weight(self):
        """
        Initialize weight using truncated normal distribution with Xavier/Glorot scaling.
        Truncation at ±3σ ensures no extreme values during initialization.
        """
        # Xavier initialization standard deviation: sqrt(2 / (fan_in + fan_out))
        std = math.sqrt(2 / (self.d_in + self.d_out))
        # Create empty tensor with desired shape and device
        empty_weight = torch.empty(size=(self.d_out, self.d_in), device=self.device, dtype=self.dtype)
        return nn.Parameter(
            # Truncated normal: values beyond ±3*std are resampled
            nn.init.trunc_normal_(empty_weight, mean=0.0, std=std, a=-3*std, b=3*std),
            requires_grad=True  # Enable gradient computation
        )
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        """
        Apply linear transformation to input tensor.
        
        Args:
            x: Input tensor of shape (..., d_in)
        
        Returns:
            Output tensor of shape (..., d_out)
        
        Example:
            假设 weight.shape = [3072, 768]  # d_out=3072, d_in=768
            x.shape = [2, 10, 768]  # batch=2, seq_len=10, d_in=768
            
            计算过程:
            1. 对每个 [768] 向量，与 weight^T 做矩阵乘法
            2. weight: [3072, 768] -> 每行是一个输出神经元的权重
            3. x[i,j,:] @ weight.T = [768] @ [768, 3072] = [3072]
            4. 输出: [2, 10, 3072]
        """
        # Linear transformation using einsum for clarity
        # Equivalent to: x @ self.weight.T
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
        empty_weight = torch.empty(size=(self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype)
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
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Normalizes input by its RMS value and applies learnable scaling.
    More efficient than LayerNorm as it doesn't center the data (no mean subtraction).
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: Dimension to normalize over (typically the model dimension)
            eps: Small constant for numerical stability (prevents division by zero)
            device: Device to place the weight tensor
            dtype: Data type of the weight tensor
        
        Example:
            d_model=768, eps=1e-5
            Creates scaling weight of shape [768] initialized to all ones
        """
        super().__init__()
        
        self.eps = eps
        self.d_model = d_model
        # Learnable scaling parameter, initialized to ones
        self.weight: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.ones(size=(d_model, ), device=device, dtype=dtype),
            requires_grad=True
        )
    
    def forward(self, x: Float[torch.Tensor, "batch_size sequence_length d_model"]) -> torch.Tensor:
        """
        Apply RMS normalization with learnable scaling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
        
        Returns:
            Normalized and scaled tensor of same shape as input
        
        Example:
            假设 x.shape = [2, 3, 4]  # batch=2, seq_len=3, d_model=4
            x[0, 0, :] = [2.0, 4.0, 6.0, 8.0]
            
            计算过程:
            1. 计算RMS: rms = sqrt(mean(x²)) = sqrt((4+16+36+64)/4) = sqrt(30) ≈ 5.48
            2. 归一化: x_norm = x / rms = [0.365, 0.730, 1.095, 1.460]
            3. 缩放: output = x_norm * weight (假设weight全为1)
            
            公式: output = (x / sqrt(mean(x²) + eps)) * weight
        """
        # Upcast to float32 for numerical stability during normalization
        x = x.to(torch.float32)
        
        # Compute Root Mean Square along d_model dimension
        rms = torch.sqrt(
                reduce(
                    x ** 2,  # Square each element
                    "... d_model -> ... 1",  # Sum over d_model, keep other dims
                    "sum",
                ) 
                / self.d_model  # Mean of squares
                + self.eps  # Add epsilon for numerical stability
        )
        
        # Normalize by RMS and apply learnable scaling
        return x / rms * self.weight

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.
    A variant of GLU that uses SiLU (Swish) as the activation function.
    More effective than ReLU/GELU for transformer feedforward networks.
    Architecture: SwiGLU(x) = (SiLU(xW1) ⊙ xW3)W2
    """
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model: Input/output dimension (model dimension)
            d_ff: Feedforward dimension (typically 4 * d_model)
            device: Device to place weight tensors
            dtype: Data type of weight tensors
        
        Example:
            d_model=768, d_ff=3072
            Creates three linear layers:
            - w1: [768 -> 3072] for gating path
            - w2: [3072 -> 768] for output projection
            - w3: [768 -> 3072] for value path
        """
        super().__init__()
        
        # Three linear transformations for SwiGLU
        self.w1 = Linear(d_model, d_ff, device, dtype)  # Gate projection
        self.w2 = Linear(d_ff, d_model, device, dtype)  # Output projection
        self.w3 = Linear(d_model, d_ff, device, dtype)  # Value projection
    
    def SiLU(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        """
        SiLU (Sigmoid Linear Unit) activation, also known as Swish.
        SiLU(x) = x * sigmoid(x)
        
        Smooth, non-monotonic activation that performs better than ReLU in deep networks.
        """
        return x * torch.sigmoid(x)
    
    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        Apply SwiGLU transformation to input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of same shape as input (..., d_model)
        
        Example:
            假设 x.shape = [2, 3, 768]  # batch=2, seq_len=3, d_model=768
            d_ff = 3072
            
            计算流程:
            1. gate = w1(x) -> [2, 3, 3072]  # 门控路径
            2. value = w3(x) -> [2, 3, 3072]  # 数值路径
            3. activated = SiLU(gate) -> [2, 3, 3072]  # 应用SiLU激活
            4. gated = activated * value -> [2, 3, 3072]  # 逐元素相乘(门控机制)
            5. output = w2(gated) -> [2, 3, 768]  # 投影回原始维度
            
            门控机制允许模型学习选择性地传递信息，提高表达能力。
        """
        return self.w2(  # Project back from d_ff to d_model
            self.SiLU(  # Apply SiLU activation to gate
                self.w1(x)  # Gate projection: d_model -> d_ff
            ) 
            *  # Element-wise multiplication (gating)
            self.w3(x)  # Value projection: d_model -> d_ff
        )

class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE) - applies rotation-based position encoding to input embeddings.
    RoPE encodes position by rotating pairs of dimensions based on their position in the sequence.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Args:
            theta: Base frequency parameter (typically 10000)
            d_k: Dimension of keys/queries (must be even)
            max_seq_len: Maximum sequence length to precompute sin/cos for
            device: Device to store precomputed values
        
        Example:
            theta=10000, d_k=4, max_seq_len=100
            Creates rotation matrices for 2 pairs of dimensions (d_k/2 = 2)
        """
        super().__init__()
        
        # d_g = d_k/2 (number of dimension pairs to rotate)
        k: Float[torch.Tensor, "d_g,"] = torch.arange(0, d_k//2, step=1.0, device=device)
        # Compute frequencies for each dimension pair: 1/(theta^(2k/d_k))
        theta_denominator: Float[torch.Tensor, "d_g,"] = 1.0 / (theta ** (2 * k / d_k))
        # Position indices from 0 to max_seq_len-1
        i: Float[torch.Tensor, "seq_len,"] = torch.arange(0, max_seq_len, step=1.0, device=device)
        # final_theta[i][j] -> rotation angle for position i, dimension pair j
        final_theta: Float[torch.Tensor, "seq_len, d_g"] = einsum(i, theta_denominator, "seq_len, d_g -> seq_len d_g")
        
        # Precompute and cache sin/cos values for efficiency
        self.register_buffer("final_theta_sin", torch.sin(final_theta), persistent=False)
        self.register_buffer("final_theta_cos", torch.cos(final_theta), persistent=False)
    
    def forward(self, 
        x: Float[torch.Tensor, "... seq_len d_k"], 
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices for each token (..., seq_len)
        
        Returns:
            Tensor with rotary position embeddings applied (..., seq_len, d_k)
        
        Example:
            假设 x.shape = [2, 3, 4]  # batch=2, seq_len=3, d_k=4
            token_positions = [[0, 1, 2], [0, 1, 2]]  # positions for each sequence
            
            对于 x[0, 0, :] = [x0, x1, x2, x3]，位置0:
            1. 分成pairs: (x0, x1), (x2, x3)
            2. 应用rotation matrix:
               [x0', x1'] = [cos(θ₀)*x0 - sin(θ₀)*x1, sin(θ₀)*x0 + cos(θ₀)*x1]
               [x2', x3'] = [cos(θ₁)*x2 - sin(θ₁)*x3, sin(θ₁)*x2 + cos(θ₁)*x3]
            3. 输出: [x0', x1', x2', x3']
            
            其中 θ₀, θ₁ 是基于位置0和维度对索引计算的角度
        """
        # Extract even and odd indexed elements (pairs of dimensions)
        # x[..., 1::2] gets indices 1, 3, 5, ... (even positions, 0-indexed)
        # x[..., 0::2] gets indices 0, 2, 4, ... (odd positions, 0-indexed)
        x_even = x[..., 1::2]  # Second element of each pair
        x_odd = x[..., 0::2]   # First element of each pair
        
        # Prepare for sin multiplication: [-x_even, x_odd, -x_even, x_odd, ...]
        # This creates the pattern needed for rotation matrix multiplication
        x_sin = torch.stack((-x_even, x_odd), dim=-1).flatten(-2)
        
        # Get precomputed sin/cos values for given positions
        # repeat_interleave(2, dim=-1) expands from d_k/2 to d_k by repeating each value twice
        # Example: [sin(θ₀), sin(θ₁)] -> [sin(θ₀), sin(θ₀), sin(θ₁), sin(θ₁)]
        sin = self.final_theta_sin[token_positions].repeat_interleave(2, dim=-1)
        cos = self.final_theta_cos[token_positions].repeat_interleave(2, dim=-1)
        
        """
        Rotation matrix for each pair (x₁, x₂):
        [x₁']   [cos(θ)  -sin(θ)] [x₁]
        [x₂'] = [sin(θ)   cos(θ)] [x₂]
        
        Expanded:
        x₁' = cos(θ)*x₁ - sin(θ)*x₂
        x₂' = sin(θ)*x₁ + cos(θ)*x₂
        
        Our implementation:
        sin_part: [-sin(θ)*x₂, sin(θ)*x₁, ...]  (handles the sin terms)
        cos_part: [cos(θ)*x₁, cos(θ)*x₂, ...]   (handles the cos terms)
        """
        sin_part = x_sin * sin
        cos_part = x * cos

        # Combine sin and cos parts to get final rotated embeddings
        return sin_part + cos_part


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    # 取出dim中最大值
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch ... queries d_k"],
    K: Float[torch.Tensor, "batch ... keys d_k"],
    V: Float[torch.Tensor, "batch ... values d_v"], # values should be the same with queries
    mask: Float[torch.Tensor, "batch ... queries keys"] | None = None
) -> Float[torch.Tensor, "batch ... queries d_v"]:
    QK = einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
    d_k = Q.shape[-1]
    denominator = math.sqrt(d_k)
    QK /= denominator
    
    # add mask
    if mask is not None:
        # mask 中 false 是需要掩码
        # 我们 reverse mask，让 false -> True, 在True填上-inf
        QK = QK.masked_fill(~mask, float("-inf"))
    
    # last dim is the key dim; ensure the query sum to 1
    scores = softmax(QK, dim=-1)
    
    attention = einsum(scores, V, "... q k, ... k d_v -> ... q d_v")
    
    return attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        use_rope: bool = False,
        max_seq_len: int = 1024,
        theta: float = 1000,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        # check if num_heads are valid
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_k = d_model // num_heads
        
        self.rope = None
        if use_rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        self.token_positions = token_positions
        
        self.q_proj = Linear(d_model, d_model) # 所有头的projection
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        batch_dim = x.shape[:-2]
        
        Q: Float[torch.Tensor, "... seq_len d_model"] = self.q_proj(x)
        K: Float[torch.Tensor, "... seq_len d_model"] = self.k_proj(x)
        V: Float[torch.Tensor, "... seq_len d_model"] = self.v_proj(x)

        # 将d_model 拆分成num_of_heads 和 d_k
        # 将heads 作为倒数第二个可以让最后seq_len d_k独立处理
        Q = rearrange(Q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        K = rearrange(K, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        V = rearrange(V, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        
        # rope
        if self.token_positions is None:
            # 1. 生成shape 是 (seq_len，)
            # 2. 增加维度 为了 匹配到 batch, seq_len 的shape
            # 3. expand 到相同 batch
            self.token_positions = torch.arange(seq_len).unsqueeze(0).expand(*batch_dim, seq_len)
        if self.rope is not None:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)
        
        # causal masking
        # 1. 生成矩阵(seq_len, seq_len) 全部都是1
        # 2. 将矩阵下三角包括对角线都设置为0
        # 3. 1->True; 0->False
        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # scaled_dot_product_attention -> True=-inf
        # 取反，将mask上三角设置为True（不包括对角线）
        # qkv = scaled_dot_product_attention(Q, K, V, ~mask)
        
        # 可以直接用tril
        # 上三角都是0，对角线不是
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        QKV = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并会d_model
        QKV = rearrange(QKV, "... heads seq_len d_k -> ... seq_len (heads d_k)")
        return self.output_proj(QKV)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta)
        
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        mha = self.attn(self.ln1(x))
        x_attention = x + mha
        return x_attention + self.ffn(self.ln2(x_attention))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, tokens: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)