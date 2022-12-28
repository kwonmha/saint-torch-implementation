
import torch
import torch.nn as nn

from models.model_utils import get_clones
from models.modules import EncoderBlock, DecoderBlock, BaseModel


class TransformerBasedModel(BaseModel):
    @classmethod
    def get_parser(cls, parser):
        parser = super(TransformerBasedModel, cls).get_parser(parser)
        parser.add_argument("--seq_len", type=int, default=100)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--att_dropout", action="store_true", default=False)
        return parser

    def __init__(self, kwargs):
        super().__init__(kwargs)


class SaintModel(TransformerBasedModel):

    @classmethod
    def get_parser(cls, parser):
        parser = super(SaintModel, cls).get_parser(parser)
        return parser

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.num_en = kwargs.n_layers
        self.num_de = kwargs.n_layers
        self.heads_en = kwargs.n_heads
        self.heads_de = kwargs.n_heads

        self.pos_embedding = nn.Embedding(self.seq_len + 1, embedding_dim=self.model_dim)  ## position

        # encoder embedding
        self.emb_ex = nn.Embedding(self.total_ex + 2, embedding_dim=self.model_dim)  # embeddings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.emb_cat = nn.Embedding(self.total_cat + 1, embedding_dim=self.model_dim)

        # decoder embedding
        # use 3 as start token(1 - incorrect, 2 - correct)
        self.emb_in = nn.Embedding(self.total_in + 2, embedding_dim=self.model_dim)  # interaction embedding

        self.dropout = nn.Dropout(self.dropout_prob)

        # for scaling output of embedding layers except positional encoding
        self.scale = torch.sqrt(torch.FloatTensor([self.model_dim]))

        self.encoder = get_clones(EncoderBlock(self.model_dim, self.n_heads, self.dropout_prob, self.norm_first, self.att_dropout),
                                  self.n_layers)
        self.decoder = get_clones(DecoderBlock(self.model_dim, self.n_heads, self.dropout_prob, self.norm_first, self.att_dropout),
                                  self.n_layers)

        if self.norm_first:
            self.final_layer_norm = nn.LayerNorm(self.model_dim)

        self.out = nn.Linear(in_features=self.model_dim, out_features=1)

    def forward(self, in_ex, in_cat, in_inter):
        device = in_ex.device

        in_ex = self.emb_ex(in_ex)
        in_cat = self.emb_cat(in_cat)

        pos_id = torch.arange(in_ex.size(1)).unsqueeze(0).to(device)
        enc_pos = self.pos_embedding(pos_id)

        # combining the embeddings
        enc_input = in_ex + in_cat + enc_pos  # (b,n,d)
        enc_input = self.dropout(enc_input)

        ## pass through each of the encoder blocks in sequence
        for x in range(self.n_layers):
            enc_output = self.encoder[x](enc_input)
            enc_input = enc_output  # passing same output as q,k,v to next encoder block

        in_inter = self.emb_in(in_inter)
        dec_pos = self.pos_embedding(pos_id)

        # combining the embeddings
        dec_input = in_inter + dec_pos  # (b,n,d)
        dec_input = self.dropout(dec_input)

        ## pass through each decoder blocks in sequence
        for x in range(self.n_layers):
            dec_output = self.decoder[x](dec_input, en_out=enc_output)
            dec_input = dec_output

        if self.norm_first:
            dec_output = self.final_layer_norm(dec_output)

        ## Output layer
        # output = torch.sigmoid(self.out(dec_output))
        output = self.out(dec_output)
        output = torch.squeeze(output)
        return output
