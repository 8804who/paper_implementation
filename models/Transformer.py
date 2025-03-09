import math

import numpy as np
from pyexpat import features


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return np.maximum(0,x)

def d_relu(x):
    return (x > 0).astype(np.float64)

class MultiHeadAttention:
    def __init__(self, heads, d_model):
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads

        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

        self.cache = {}

    def split_heads(self, X):
        batch_size, seq_length, d_model = X.shape
        X = X.reshape(batch_size, seq_length, self.heads, self.d_k)
        return X.transpose(0,2,1,3)

    def combine_heads(self, X):
        batch_size, num_heads, seq_length, depth = X.shape
        X = X.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        return X

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.maksed_fill(mask == 0, -1e9)
        attn_weights = softmax(attn_scores)
        output = np.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['mask'] = mask

        q = np.matmul(Q, self.W_q)
        k = np.matmul(K, self.W_k)
        v = np.matmul(V, self.W_v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        self.cache['q_proj'] = q
        self.cache['k_proj'] = k
        self.cache['v_proj'] = v

        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        self.cache['attention_weights'] = attention_weights

        attention_output = self.combine_heads(attention_output)

        output = np.matmul(attention_output, self.W_o)
        self.cache['attention_output'] = attention_output

        return output

    def backward(self, d_out):
        batch_size = d_out.shape[0]

        self.dW_o += np.matmul(self.cache['attention_output'].transpose(1, 2))
        d_attention_output = np.matmul(d_out, self.W_o.T).reshape(batch_size, -1, self.heads, self.d_k).transpose(1,2)

        q_proj = self.cache['q_proj']
        k_proj = self.cache['k_proj']
        v_proj = self.cache['v_proj']
        attention_weights = self.cache['attention_weights']

        d_v = np.matmul(attention_weights.transpose(0,1,3,2), d_attention_output)

        d_attention_weights = np.matmul(d_attention_output, v_proj.transpose(0,1,3,2))

        d_scores = d_attention_weights * (attention_weights * (1 - attention_weights))

        d_q = np.matmul(d_scores, k_proj) / np.sqrt(self.d_k)
        d_k = np.matmul(d_scores.transpose(0, 1, 3, 2), q_proj) / np.sqrt(self.d_k)

        d_q = d_q.transpose(0, 2, 1, 3).reshape(batch_size, -1 ,self.d_model)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(batch_size, -1 ,self.d_model)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(batch_size, -1 ,self.d_model)

        q = self.cache['q']
        k = self.cache['k']
        v = self.cache['v']

        self.dW_q += np.matmul(q.transpose(0, 2, 1), d_q).sum(0)
        self.dW_k += np.matmul(k.transpose(0, 2, 1), d_k).sum(0)
        self.dW_v += np.matmul(v.transpose(0, 2, 1), d_v).sum(0)

        d_query = np.matmul(d_q, self.W_q.T)
        d_key = np.matmul(d_k, self.W_k.T)
        d_value = np.matmul(d_v, self.W_v.T)

        return d_query, d_key, d_value

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.normalized = (x-self.mean) / np.sqrt(self.var+self.eps)
        return self.gamma * self.normalized + self.beta

    def backward(self, d_out):
        self.d_gamma = np.sum(d_out * self.normalized, axis=(0, 1))
        self.d_beta = np.sum(d_out, axis=(0, 1))

        d_normalized = d_out * self.gamma

        d_var = np.sum(d_normalized * (self.x - self.mean))
        d_mean = np.sum(d_normalized * -1 / np.sqrt(self.var+self.eps))

        d_x = d_normalized / np.sqrt(self.var + self.eps) + d_var * 2 * (self.x - self.mean) / features + d_mean / features
        return d_x

class PositionWiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff =d_ff

        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

        self.d_W1 = np.zeros_like(self.W1)
        self.d_b1 = np.zeros_like(self.b1)
        self.d_W2 = np.zeros_like(self.W2)
        self.d_b2 = np.zeros_like(self.b2)

        self.cache = {}

    def forward(self, x):
        h = np.matmul(x, self.W1)+self.b1
        h_relu = relu(h)

        self.cache['x'] = x
        self.cache['h'] = h
        self.cache['h_relu']= h_relu

        return np.matmul(h_relu, self.W2)+self.b2

    def backward(self, d_out):
        x = self.cache['x']
        h = self.cache['h']
        h_relu = self.cache['h_relu']

        self.d_W2 += np.matmul(h_relu.reshape(-1, self.d_ff).T, d_out.reshape(-1, self.d_model))
        self.d_b2 += np.sum(d_out, axis=(0, 1))

        d_h_relu = np.matmul(d_out, self.W2.T)

        d_h = d_h_relu * d_relu(h)

        self.d_W1 += np.matmul(x.reshape(-1, self.d_model).T, d_h.reshape(-1, self.d_ff))
        self.d_b1 += np.sum(d_h, axis=(0,1))

        d_x = np.matmul(d_h, self.W1.T)

        return d_x

class Encoder:
    def __init__(self, d_model, d_ff):
        self.Q = np.random.randn(d_model, d_model) * 0.01
        self.K = np.random.randn(d_model, d_model) * 0.01
        self.V = np.random.randn(d_model, d_model) * 0.01

        self.FeedForwardNetwork = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)


class Decoder:
    def __init__(self, d_model, d_ff):
        self.Q = np.random.randn(d_model, d_model) * 0.01
        self.K = np.random.randn(d_model, d_model) * 0.01
        self.V = np.random.randn(d_model, d_model) * 0.01

        self.FeedForwardNetwork = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)


class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_seq_length=5000):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.input_embeddings = np.random.randn(src_vocab_size, d_model)
        self.output_embeddings = np.random.randn(tgt_vocab_size, d_model)
        self.encoders = [Encoder(self.d_model, self.d_ff) for _ in range(num_layers)]
        self.decoders = [Decoder(self.d_model, self.d_ff) for _ in range(num_layers)]
        self.pe = self.init_positional_encoding(max_seq_length)

    def init_positional_encoding(self, max_seq_length):
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0)/self.d_model))
        pe = np.zeros((max_seq_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def encode(self, x):
        X_emb = self.embeddings[x]+self.pe[x]
        for layer in self.encoders:
            X_emb = layer
        return X_emb
