import torch
import torch.nn
from utils import Config


class MLP(torch.nn.Module):
    def __init__(self, embed, out):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(embed, embed*3)
        self.fc2 = torch.nn.Linear(embed*3, embed*9)
        self.fc3 = torch.nn.Linear(embed*9, out)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, config:Config):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=config.embed_dim,
                                                          nhead=config.head_num,
                                                          dim_feedforward=config.embed_dim,
                                                            dropout=config.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=config.layer_num,
                                                               norm=torch.nn.LayerNorm(config.embed_dim))
        self.transformer_encoder_first = torch.nn.TransformerEncoder(encoder_layers, num_layers=4,
                                                                     norm=torch.nn.LayerNorm(config.embed_dim))

        self.patch_embedding = torch.nn.Linear(config.patch_size, config.embed_dim)
        self.config = config
        self.device = 'cuda:1'
        # learnable token
        self.cls_token = torch.nn.Parameter(torch.randn(1, config.instance_num, config.embed_dim))
        self.mlps = torch.nn.ModuleList([MLP(config.embed_dim, config.cls_num) for _ in range(config.instance_num)])
        self.to(self.device)

    def spectra_window(self, spectra):
        # spectra: [batch, signal_length]->[batch, seq_len, embed_dim]
        batch_size = spectra.shape[0]
        spectra = spectra.view(batch_size, -1, self.config.patch_size)
        return spectra

    def forward(self, spectra):
        # spectra: [batch, signal_length]
        spectra = self.spectra_window(spectra)
        # spectra: [batch, seq_len, embed_dim]
        spectra = self.patch_embedding(spectra)
        spectra = self.position_encoding(spectra)+spectra

        # spectra: [batch, seq_len, embed_dim]
        spectra = self.transformer_encoder_first(spectra)

        spectra = torch.concat([self.cls_token.expand(spectra.shape[0], -1, -1), spectra], dim=1)
        spectra = self.transformer_encoder(spectra)
        cls_token = spectra[:, :self.config.instance_num, :]
        output = torch.stack([mlp(cls_token[:, i, :]) for i, mlp in enumerate(self.mlps)], dim=1)
        # print(cls.shape)
        return output

    def load_checkpoint(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()

    def position_encoding(self, spectra):
        seq_len = spectra.shape[1]
        n_embd = spectra.shape[2]
        pe = torch.zeros(seq_len, n_embd, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


if __name__ == '__main__':
    config = Config()
    model = Transformer(config)
    spectra = torch.randn(10, 512).to(model.device)
    label = torch.randint(0, 2, (10,37)).to(model.device)
    # onehot
    label = torch.nn.functional.one_hot(label, 2).float().reshape(-1,2)
    print(label.shape)
    output = model(spectra)
    print(output.shape)
    # loss
    loss = torch.nn.CrossEntropyLoss()(output.reshape(-1,2), label)
    print(loss)
    loss = torch.nn.CrossEntropyLoss()(label, label)
    print(loss)