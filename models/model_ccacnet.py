import torch.nn as nn

from models.ccacnet.model_full import CrossViewModel


class CCACNetClassifier(nn.Module):
    """
    Attack-side wrapper for CCAC-Net.

    The wrapped model keeps CCAC-Net's original state_dict keys usable while
    matching the other migrated model wrappers in this project.
    """

    def __init__(self, num_actions):
        super().__init__()
        self.model = CrossViewModel(num_actions)

    @property
    def num_actions(self):
        return self.model.num_actions

    @property
    def use_auxiliary_modules(self):
        return self.model.use_auxiliary_modules

    def set_auxiliary_modules_enabled(self, enabled):
        self.model.set_auxiliary_modules_enabled(enabled)

    def forward(self, inputs):
        return self.model(inputs)
