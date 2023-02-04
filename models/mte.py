import torch
from torch import nn, Tensor
import math
from einops import repeat


class MTE(nn.Module):

    def __init__(self, m1, m2, n):
        super().__init__()
        self.mod1 = nn.Sequential()
        self.mod1.add_module('mod1_fc1', nn.Linear(m1, 2048))
        # self.mod1.add_module('mod1_bn1', nn.BatchNorm1d(2048))
        self.mod1.add_module('mod1_relu1', nn.ReLU(inplace=True))
        self.mod1.add_module('mod1_fc2', nn.Linear(2048, 2048))
        # self.mod1.add_module('mod1_bn2', nn.BatchNorm1d(1024))
        self.mod1.add_module('mod1_relu2', nn.ReLU(inplace=True))
        self.mod1.add_module('mod1_fc3', nn.Linear(2048, 1024))
        # self.mod1.add_module('mod1_bn3', nn.BatchNorm1d(512))
        self.mod1.add_module('mod1_relu3', nn.ReLU(inplace=True))

        self.mod2 = nn.Sequential()
        self.mod2.add_module('mod2_fc1', nn.Linear(m2, 256))
        # self.mod2.add_module('mod2_bn1', nn.BatchNorm1d(128))
        self.mod2.add_module('mod2_relu', nn.ReLU(inplace=True))
        self.mod2.add_module('mod2_fc2', nn.Linear(256, 512))
        # self.mod2.add_module('mod2_bn2', nn.BatchNorm1d(64))
        self.mod2.add_module('mod2_relu2', nn.ReLU(inplace=True))
        self.mod2.add_module('mod2_fc3', nn.Linear(512, 256))
        self.mod2.add_module('mod2_relu3', nn.ReLU(inplace=True))

        tel = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True, norm_first=True)
        self.t = nn.TransformerEncoder(tel, num_layers=6)

        self.reg = nn.Sequential()
        self.reg.add_module('reg_fc1', nn.Linear(256 * 1024, 1024))
        # self.reg.add_module('reg_bn1', nn.BatchNorm1d(1024))
        self.reg.add_module('reg_relu1', nn.ReLU(inplace=True))
        self.reg.add_module('reg_fc2', nn.Linear(1024, 512))
        # self.reg.add_module('reg_bn2', nn.BatchNorm1d(512))
        self.reg.add_module('reg_relu2', nn.ReLU(inplace=True))
        self.reg.add_module('reg_fc3', nn.Linear(512, n))

    def forward(self, x1, x2):
        modularity1 = self.mod1(x1)
        modularity2 = self.mod2(x2)
        modularity1 = torch.reshape(modularity1, (-1, 1, modularity1.shape[-1]))
        modularity2 = torch.reshape(modularity2, (-1, 1, modularity2.shape[-1]))
        features = torch.matmul(torch.transpose(modularity2, dim0=1, dim1=2), modularity1)
        t_o = self.t(features)
        t_o = torch.reshape(t_o, (-1, modularity1.shape[-1] * modularity2.shape[-1]))
        output = self.reg(t_o)

        del modularity1, modularity2, features, t_o

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MMTE(nn.Module):

    # d_model = 12
    def __init__(self, f1, f2, d_model, n):
        super(MMTE, self).__init__()
        self.d_model = d_model
        self.seq_len1 = 1000
        self.seq_len2 = 20

        self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
        self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)

        self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
        self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))
        # self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)

        encoder_layers1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
        encoder_layers2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=6, activation='gelu', batch_first=True)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers1, num_layers=6)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers2, num_layers=6)

        # mmtemlp
        # self.decoder = nn.Sequential()
        # self.decoder.add_module('fc1', nn.Linear(d_model, 1024))
        # self.decoder.add_module('relu1', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc2', nn.Linear(1024, 512))
        # self.decoder.add_module('relu2', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc3', nn.Linear(512, 128))
        # self.decoder.add_module('relu3', nn.ReLU(inplace=True))
        # self.decoder.add_module('fc4', nn.Linear(128, 128))
        # self.decoder.add_module('fc5', nn.Linear(128, n))

        # mte
        self.decoder = nn.Linear(d_model, n)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        lx1 = self.lp_m1(x1)
        lx2 = self.lp_m2(x2)
        lx1 = torch.reshape(lx1, (batch_size, self.seq_len1, self.d_model))
        lx2 = torch.reshape(lx2, (batch_size, self.seq_len2, self.d_model))

        # lx1 = self.pos_encoder(lx1)
        # lx2 = self.pos_encoder(lx2)
        cls_tok1 = repeat(self.cls1, '() n e -> b n e', b=batch_size)
        cls_tok2 = repeat(self.cls2, '() n e -> b n e', b=batch_size)
        lx1 = torch.cat([cls_tok1, lx1], dim=1)
        lx2 = torch.cat([cls_tok2, lx2], dim=1)
        lx1 += self.positions1
        lx2 += self.positions2

        output1 = self.transformer_encoder1(lx1)
        output2 = self.transformer_encoder2(lx2)

        # output1 = torch.mean(output1, 1, True)
        # output2 = torch.mean(output2, 1, True)
        # decoder_in = torch.add(output1, output2)
        decoder_in = torch.add(output1[:, 0, :], output2[:, 0, :])
        output = self.decoder(decoder_in.reshape(batch_size, self.d_model))

        del lx1, lx2, output1, output2, decoder_in

        return output


class MMTE2(nn.Module):

    def __init__(self, f, d_model, n):
        super(MMTE2, self).__init__()
        self.d_model = d_model
        self.seq_len1 = 1000
        self.seq_len2 = 20

        self.lp_m1 = nn.Linear(f1, self.seq_len1 * d_model)
        self.lp_m2 = nn.Linear(f2, self.seq_len2 * d_model)

        self.cls1 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.cls2 = nn.Parameter(torch.rand(1, 1, self.d_model))
        self.positions1 = nn.Parameter(torch.rand(self.seq_len1 + 1, d_model))
        self.positions2 = nn.Parameter(torch.rand(self.seq_len2 + 1, d_model))
