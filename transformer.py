"""
@Author: danyzhou
@Updated: 9/19/25 3:52 PM

Advantest Confidential - All Rights Reserved

手写 Transformer（Encoder-Decoder）实现，带逐行注释（中文）。
适合学习与小规模实验，不是最优性能实现。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Scaled Dot-Product Attention
# -------------------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    计算缩放点积注意力 (Scaled Dot-Product Attention)
    q, k, v: 张量形状分别为 (batch, n_head, seq_len, head_dim)
    mask: 可选的 mask，形状兼容 (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
    返回: attention 输出，和 attention 权重（可选）
    """
    # q @ k^T：注意力分数，形状 -> (batch, n_head, seq_q, seq_k)
    # 注意：使用 transpose(-2, -1) 把 k 的最后两个维度转置以做矩阵乘法
    scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L_q, L_k]

    # 缩放：除以 sqrt(d_k)（防止内积值过大导致 softmax 梯度消失）
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)

    # 应用 mask（如果提供）。mask 中通常用 0 表示保留，-inf 或 False 表示屏蔽
    if mask is not None:
        # 假设 mask 中被屏蔽的位置是 0，保留位置是 1（布尔或 0/1）
        # 为被 mask 的位置加上很小的值 -1e9（在 softmax 后近似为 0）
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # softmax 得到注意力权重
    attn = torch.softmax(scores, dim=-1)  # [B, H, L_q, L_k]

    # dropout（可选）
    if dropout is not None:
        attn = dropout(attn)

    # 权重 * v -> 输出
    output = torch.matmul(attn, v)  # [B, H, L_q, head_dim]
    return output, attn


# -------------------------
# 2) Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """
        d_model: 模型隐藏维度（例如 512）
        n_head: 多头数量（例如 8）
        head_dim = d_model / n_head 必须为整数
        """
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 用线性映射把输入映射成 Q, K, V（一次性做，然后 split heads）
        # 我们用一个 Linear 做 qkv 的投影，也可以分别使用三个 Linear。
        # 这里使用 separate 线性层以便更直观。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 最后的线性层：把多头输出 concat 后映射回 d_model
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: [batch, seq_len, d_model]
        mask: broadcastable 到 [batch, n_head, seq_len_q, seq_len_k] 的形状
        返回: [batch, seq_len, d_model]
        """
        B = query.size(0)

        # 1) 线性投影 -> 得到 Q, K, V：形状 [B, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2) 切分为 heads：先 reshape 再 transpose
        # 先把最后维度 d_model 拆成 (n_head, head_dim)
        # reshape -> [B, seq_len, n_head, head_dim]
        # transpose -> [B, n_head, seq_len, head_dim]
        def shape_for_heads(x):
            return x.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        Q = shape_for_heads(Q)
        K = shape_for_heads(K)
        V = shape_for_heads(V)

        # 3) 如果 mask 给到的是 [B, 1, 1, seq_k] 或 [B, 1, seq_q, seq_k] 之类的，
        #    我们需要把它扩展到 [B, n_head, seq_q, seq_k]，这里通过广播实现。
        if mask is not None:
            # 假设 mask 形状是 [B, 1, 1, seq_k] 或 [B, 1, seq_q, seq_k]
            # 不需要显式 expand，这里确保 mask 的类型是 bool 或 0/1
            pass

        # 4) 调用缩放点积注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 5) 把多头输出 concat 回原始形状
        # attn_output 形状 [B, n_head, seq_len, head_dim]
        # transpose -> [B, seq_len, n_head, head_dim]
        # contiguous + view -> [B, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 6) 最后的线性映射
        out = self.fc_out(attn_output)  # [B, seq_len, d_model]
        return out, attn_weights


# -------------------------
# 3) Position-wise Feed-Forward Network
# -------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈网络：两个线性层 + 非线性（ReLU） + dropout
        d_ff: feed-forward 隐藏层维度（通常比 d_model 大 4 倍）
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 也可用 GELU

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # 返回形状 [B, seq_len, d_model]


# -------------------------
# 4) Positional Encoding（位置编码）
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        采用 Vaswani 提出的正弦/余弦位置编码（可加到 embedding 上）
        该编码是固定的（非学习），但也可以替换为 nn.Parameter 学习式编码
        """
        super().__init__()
        # 初始化一个 (max_len, d_model) 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # 计算分母项（不同 dimension 用不同频率）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置使用 sin，奇数位置使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        # 增加 batch 维度 -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册为 buffer（不是参数，不会被优化器更新，但会随模型一起保存/移动设备）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        返回 x + pos_encoding[:seq_len]
        """
        seq_len = x.size(1)
        # self.pe[:, :seq_len, :] -> [1, seq_len, d_model]，通过广播加到 x 上
        return x + self.pe[:, :seq_len, :].to(x.device)


# -------------------------
# 5) Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力子层（自注意力）
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        # 前馈子层
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        # 残差连接 + 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        x: [B, src_len, d_model]
        src_mask: mask for attention, broadcastable to [B, n_head, src_len, src_len]
        """
        # 自注意力子层
        # 注意：为了保持维度一致，先做 LayerNorm，再做 attention（"Pre-LN" 结构）
        # 也可以用传统 "Post-LN"：先 attention -> add & norm
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2, mask=src_mask)  # query=key=value=x
        x = x + self.dropout(attn_out)  # 残差连接

        # 前馈子层
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)  # 残差连接
        return x


# -------------------------
# 6) Decoder Layer
# -------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)  # masked self-attn
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)  # encoder-decoder attn
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        x: [B, tgt_len, d_model] (decoder input embeddings)
        enc_output: [B, src_len, d_model] (encoder output)
        tgt_mask: 用于 decoder 自注意力的 mask（通常包含 future mask & pad mask）
        memory_mask: 用于 encoder->decoder attention 的 mask（例如屏蔽 encoder 的 padding）
        """
        # 1) masked self-attention（防止看到未来位置）
        x2 = self.norm1(x)
        self_attn_out, _ = self.self_attn(x2, x2, x2, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)

        # 2) encoder-decoder cross attention
        x2 = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(x2, enc_output, enc_output, mask=memory_mask)
        x = x + self.dropout(cross_attn_out)

        # 3) 前馈
        x2 = self.norm3(x)
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)
        return x


# -------------------------
# 7) Encoder & Decoder（堆叠 Layer）
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        # 多层 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        # 最后一个 LayerNorm（常用的 Pre-LN 结构会在 Layer 内进行 norm，这里再加一个全局 norm）
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_tokens, src_mask=None):
        """
        src_tokens: [B, src_len]（token ids）
        src_mask: [B, 1, 1, src_len] 或 [B, 1, src_len, src_len]，用于屏蔽 padding
        """
        # 1) token -> embedding 并缩放（原论文用 sqrt(d_model) 进行缩放）
        x = self.embedding(src_tokens) * math.sqrt(self.d_model)  # [B, src_len, d_model]
        # 2) 加位置编码
        x = self.pos_enc(x)

        # 3) 逐层编码
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        x = self.norm(x)
        return x  # [B, src_len, d_model]


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt_tokens, enc_output, tgt_mask=None, memory_mask=None):
        """
        tgt_tokens: [B, tgt_len]
        enc_output: [B, src_len, d_model]
        tgt_mask: 用于 decoder 自注意力（包含未来屏蔽与 padding）
        memory_mask: 用于 encoder-decoder attention（屏蔽 encoder 的 padding）
        """
        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)
        return x  # [B, tgt_len, d_model]


# -------------------------
# 8) Full Transformer
# -------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_head=8, d_ff=2048, num_encoder_layers=6,
                 num_decoder_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_head, d_ff, num_encoder_layers, max_len=max_len, dropout=dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_head, d_ff, num_decoder_layers, max_len=max_len, dropout=dropout)
        # 输出投影到目标词表大小（用于计算 logits）
        self.output_fc = nn.Linear(d_model, tgt_vocab)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src_tokens: [B, src_len]
        tgt_tokens: [B, tgt_len]
        返回: logits [B, tgt_len, tgt_vocab]
        """
        enc_output = self.encoder(src_tokens, src_mask=src_mask)
        dec_output = self.decoder(tgt_tokens, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.output_fc(dec_output)
        return logits


# -------------------------
# 9) Mask helpers
# -------------------------
def make_src_mask(src_tokens, pad_idx=0):
    """
    生成 encoder padding mask
    src_tokens: [B, src_len]
    返回 mask 形状 [B, 1, 1, src_len]，其中 1 表示保留，0 表示被 mask（padding）
    便于 broadcast 到 attention 分数 [B, n_head, seq_q, seq_k]
    """
    mask = (src_tokens != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,src_len]
    return mask  # 布尔或 0/1


def make_tgt_mask(tgt_tokens, pad_idx=0):
    """
    生成 decoder 的 mask：包含两部分
    1) padding mask（同 src）
    2) future mask（下三角）用于屏蔽未来位置（防止 decoder 看到后续 token）
    返回形状 [B, 1, tgt_len, tgt_len]
    """
    B, tgt_len = tgt_tokens.size()
    # padding mask
    pad_mask = (tgt_tokens != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,tgt_len]
    # subsequent mask: 下三角矩阵（1 表示允许 attention，0 表示屏蔽未来）
    subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=tgt_tokens.device))
    # 组合
    subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # [1,1,tgt_len,tgt_len]
    mask = pad_mask & (subsequent_mask.bool())  # broadcast -> [B,1,tgt_len,tgt_len]
    return mask


# -------------------------
# 10) 小测试：随机输入前向运行
# -------------------------
if __name__ == "__main__":
    # 超参数（小规模）
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    D_MODEL = 128
    N_HEAD = 8
    D_FF = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 2
    BATCH = 2
    SRC_LEN = 10
    TGT_LEN = 12
    PAD_IDX = 0

    # 创建模型
    model = Transformer(SRC_VOCAB, TGT_VOCAB, d_model=D_MODEL, n_head=N_HEAD, d_ff=D_FF,
                        num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS, max_len=50)
    model.eval()

    # 随机示例输入（整数 token id）
    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))  # 不包含 pad
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN))

    # 假设最后一位是 pad（示范），我们把最后一列设为 0（pad）
    src[:, -1] = PAD_IDX
    tgt[:, -1] = PAD_IDX

    # 构造 masks
    src_mask = make_src_mask(src, pad_idx=PAD_IDX)  # [B,1,1,SRC_LEN]
    tgt_mask = make_tgt_mask(tgt, pad_idx=PAD_IDX)  # [B,1,TGT_LEN,TGT_LEN]
    memory_mask = src_mask  # encoder-decoder attention 不关注 src 的 pad

    # 前向
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
    # logits 形状 -> [B, TGT_LEN, TGT_VOCAB]
    print("logits shape:", logits.shape)  # 期望 (BATCH, TGT_LEN, TGT_VOCAB)

    # 用 CrossEntropyLoss 做示范（注意要把 logits 展平）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # 目标 target 为下一个 token（teacher forcing 情况），这里只随意示范
    # 假设训练时输入的 tgt 是带 <sos> 的，target 是对应的下一个 token
    target = torch.randint(0, TGT_VOCAB, (BATCH, TGT_LEN))
    loss = criterion(logits.view(-1, TGT_VOCAB), target.view(-1))
    print("dummy loss:", loss.item())

