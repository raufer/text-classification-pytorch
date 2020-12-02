import torch

from transformers import RobertaModel


class ROBERTAClassifier(torch.nn.Module):

    """
    During training the output of RoBERTa is a batch of hidden states, which is passed to classifier layers
    """

    def __init__(self, dropout_rate=0.3, n_outputs=2):
        super(ROBERTAClassifier, self).__init__()
        print('DROPOUT RATE', type(dropout_rate), dropout_rate)

        self.pretrained_model = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, n_outputs)

    def forward(self, *args, **kwargs):
        _, x = self.pretrained_model(*args, **kwargs)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)

        return x