import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, q, k, v):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_weights = nn.Softmax(dim=-1)(attn_score)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model//n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(attn)
        return output, attn_weights

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        attn_outputs, attn_weights = self.mha(q, k, v)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(q + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        
        return ffn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, p_drop=0, d_ff=2048):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

            if m.bias is not None:
                init.constant_(m.bias.data, 0)

    def forward(self, feature, center):
        feature = feature.unsqueeze(1)
        center = center.unsqueeze(0)
        for layer in self.layers:
            center, attn_weights = layer(center, feature, feature)
        return center.squeeze(0)

# encoder = TransformerEncoder(n_heads=8)
# feature = torch.rand(15, 512)
# center = torch.rand(10, 512)
# print(center)
# ans = encoder(feature, center)
# print(ans.shape)
# print(ans)
