
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import get_mask


class BaseModel(nn.Module):
    @classmethod
    def get_parser(cls, parser):
        parser.add_argument("--model_dim", type=int, default=256)
        parser.add_argument("--n_layers", type=int, default=4)
        parser.add_argument("--dropout_prob", type=float, default=0.1)
        return parser

    def __init__(self, kwargs):
        super().__init__()
        kwargs = vars(kwargs)
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])


class FeedForwardLayer(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff, dropout_prob):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff * 4)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(in_features=dim_ff * 4, out_features=dim_ff)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)


class EncoderBlock(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, model_dim, heads_en, dropout_prob, norm_first, att_dropout):
        super().__init__()
        self.model_dim = model_dim
        self.norm_first = norm_first
        self.att_dropout = att_dropout

        if self.att_dropout:
            self.multi_en = nn.MultiheadAttention(model_dim, heads_en, dropout=dropout_prob)
        else:
            self.multi_en = nn.MultiheadAttention(model_dim, heads_en)  # multihead attention
        self.ffn_en = FeedForwardLayer(model_dim, dropout_prob)  # feedforward block
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, input):
        device = input.device

        out = input.permute(1, 0, 2)  # -->(n,b,d)  # print('pre multi', out.shape )

        # Multihead attention
        skip_out = out
        n, _, _ = out.shape

        if self.norm_first:
            out = self.layer_norm1(out)
            out, attn_wt = self.multi_en(out, out, out,
                                         attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out

            skip_out = out
            out = self.layer_norm2(out)
            out = self.ffn_en(out)
            out = self.dropout(out) + skip_out

        else:
            out, attn_wt = self.multi_en(out, out, out,
                                         attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out  # skip connection
            out = self.layer_norm1(out)  # Layer norm

            # feed forward
            out = out.permute(1, 0, 2)  # -->(b,n,d)
            skip_out = out
            out = self.ffn_en(out)
            out = self.dropout(out) + skip_out  # skip connection
            out = self.layer_norm2(out)  # Layer norm

        return out


class DecoderBlock(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, model_dim, heads_de, dropout_prob, norm_first, att_dropout):
        super().__init__()
        self.norm_first = norm_first
        self.att_dropout = att_dropout

        if self.att_dropout:
            self.multi_de1 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads_de,
                                                   dropout=dropout_prob)  # M1 multihead for interaction embedding as q k v
            self.multi_de2 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads_de,
                                                   dropout=dropout_prob)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        else:
            self.multi_de1 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads_de)  # M1 multihead for interaction embedding as q k v
            self.multi_de2 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads_de)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en = FeedForwardLayer(model_dim, dropout_prob)  # feed forward layer

        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

    def forward(self, dec_in, en_out):
        device = dec_in.device

        out = dec_in.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)# print('pre multi', out.shape )

        # Multihead attention M1
        n, _, _ = out.shape
        skip_out = out

        if self.norm_first:
            out = self.layer_norm1(out)
            out, attn_wt = self.multi_de1(out, out, out,
                                         attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out

            en_out = en_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
            skip_out = out
            out = self.layer_norm1(out)
            out, attn_wt = self.multi_de2(out, en_out, en_out,
                                          attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out

            out = out.permute(1, 0, 2)  # (n,b,d)-->(b,n,d)
            skip_out = out
            out = self.layer_norm3(out)
            out = self.ffn_en(out)
            out = self.dropout(out) + skip_out

        else:
            out, attn_wt = self.multi_de1(out, out, out,
                                          attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out  # skip connection
            out = self.layer_norm1(out)

            # Multihead attention M2
            en_out = en_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
            skip_out = out
            out, attn_wt = self.multi_de2(out, en_out, en_out,
                                          attn_mask=get_mask(n, device))  # attention mask upper triangular
            out = self.dropout(out) + skip_out  # skip connection
            out = self.layer_norm2(out)

            # feed forward
            out = out.permute(1, 0, 2)  # (n,b,d)-->(b,n,d)
            skip_out = out
            out = self.ffn_en(out)
            out = self.dropout(out) + skip_out  # skip connection
            out = self.layer_norm3(out)  # Layer norm

        return out
