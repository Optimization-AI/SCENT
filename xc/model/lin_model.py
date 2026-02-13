import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear classifier: maps feature vectors to logits for m classes.

    forward(x): expects shape (B, D) and returns logits (B, m)
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feature_dim, num_classes, bias=False)

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor | str | None = None) -> torch.Tensor:
        """Compute logits for only the classes in `labels`.

        Args:
            x: tensor (B, D)
            labels: 1D LongTensor of class indices (B,)
            classes: sampled classes, or 'all' (meaning all classes), or None (meaning classes in labels)

        Returns:
            logits (B, K)
        """
        w_pos = self.fc.weight[labels]  # (B, D)

        mask = None
        if isinstance(classes, str):
            assert classes == "all"
            w_sampled = self.fc.weight
        elif isinstance(classes, torch.Tensor):
            w_sampled = self.fc.weight[classes]
        elif classes is None:
            unique_labels = torch.unique(labels)    # (K,)
            w_sampled = self.fc.weight[unique_labels]    # (K, D)
            mask = (labels.unsqueeze(1) == unique_labels.unsqueeze(0))    # (B, K)
        else:
            raise ValueError(f"Unknown classes type: {type(classes)}")
        logits = x @ w_sampled.T - torch.sum(torch.mul(x, w_pos), dim=1, keepdim=True)    # (B, K)
        if mask is not None:
            logits.masked_fill_(mask.to(logits.device), float("-inf"))
        return logits
