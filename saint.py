import numpy as np
import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, state_size, dropout_prob):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAINTModel(nn.Module):
    def __init__(self, embed_dim, n_layers, dropout_prob, max_seq, n_skill, n_part):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part

        self.e_embedding = nn.Embedding(self.n_skill + 1, embed_dim)  ## exercise
        self.c_embedding = nn.Embedding(self.n_cat + 1, embed_dim)  ## category
        self.pos_embedding = nn.Embedding(max_seq + 1, embed_dim)  ## position
        self.res_embedding = nn.Embedding(2 + 1, embed_dim)  ## response

        self.transformer = nn.Transformer(nhead=8, d_model=embed_dim, num_encoder_layers=n_layers,
                                          num_decoder_layers=n_layers, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, dropout_prob)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, question, part, response):

        device = question.device

        question = self.e_embedding(question)
        part = self.c_embedding(part)
        pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)
        res = self.res_embedding(response)

        enc = question + part + pos_id
        dec = pos_id + res

        enc = enc.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask=mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)