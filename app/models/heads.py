from torch import nn


class ClassificationHead(nn.Module):
    """Simple dropout + linear layer head for classification tasks."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):  # type: ignore
        return self.classifier(self.dropout(x)) 