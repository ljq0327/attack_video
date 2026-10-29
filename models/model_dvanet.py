import torch
import torch.nn as nn

from models.dvanet.v3_backbone import VideoTransformer


class DVANETClassifier(nn.Module):
    def __init__(self, num_actions, num_views=3, dataset=None):
        super().__init__()
        self.num_actions = num_actions
        self.num_views = num_views
        self.dataset = dataset
        self.model = VideoTransformer(224, num_views, num_actions)
        self.register_buffer(
            "logit_index",
            torch.tensor(_build_logit_index(dataset, num_actions), dtype=torch.long),
            persistent=False,
        )

    def _canonicalize_logits(self, logits):
        return logits.index_select(-1, self.logit_index.to(logits.device))

    def forward_single_view(self, x):
        # x: [B, T, C, H, W]
        _, action_logits, *_ = self.model(x)
        return self._canonicalize_logits(action_logits)

    def forward(self, inputs):
        # inputs: [B, V, T, C, H, W]
        if inputs.dim() == 5:
            logits = self.forward_single_view(inputs)
            return {"per_view_logits": logits.unsqueeze(1)}

        B, V, T, C, H, W = inputs.shape
        x = inputs.contiguous().view(B * V, T, C, H, W)
        logits = self.forward_single_view(x)
        logits = logits.view(B, V, -1)

        return {"per_view_logits": logits}


def _build_logit_index(dataset, num_actions):
    canonical_actions = _canonical_actions(dataset, num_actions)
    dvanet_actions = sorted(canonical_actions, key=lambda action: str(action))
    return [dvanet_actions.index(action) for action in canonical_actions]


def _canonical_actions(dataset, num_actions):
    if dataset == "numa":
        return [1, 2, 3, 4, 5, 6, 8, 9, 11, 12]
    return list(range(1, num_actions + 1))
