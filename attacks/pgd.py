import torch
import torch.nn.functional as F

from .temporal_translation import DEFAULT_MEAN, DEFAULT_STD, ProjectVideoModel, denormalize_video, normalize_video


class ProjectedGradientDescent:
    """L-infinity PGD attack in pixel space for project video models."""

    def __init__(
        self,
        model,
        epsilon=8 / 255,
        steps=10,
        pgd_step_size=None,
        pgd_random_start=False,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        targeted=False,
        target_map_function=None,
    ):
        self.model = ProjectVideoModel(model)
        self.training = model.training
        self.device = next(model.parameters()).device
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(pgd_step_size) if pgd_step_size is not None else self.epsilon / max(self.steps, 1)
        self.random_start = bool(pgd_random_start)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.targeted = bool(targeted)
        self.target_map_function = target_map_function

        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if self.step_size < 0:
            raise ValueError(f"pgd_step_size must be >= 0, got {pgd_step_size}")

    def _switch_model_back(self):
        if self.training:
            self.model.model.train()
        else:
            self.model.model.eval()

    def _labels_for_loss(self, videos, labels):
        if not self.targeted:
            return labels
        if self.target_map_function is None:
            raise ValueError("target_map_function is required for targeted attacks.")
        return self.target_map_function(videos, labels)

    def _loss(self, logits, labels):
        cost = F.cross_entropy(logits, labels)
        return -cost if self.targeted else cost

    def __call__(self, videos, labels):
        self.model.model.eval()
        try:
            return self.forward(videos, labels)
        finally:
            self._switch_model_back()

    def forward(self, videos, labels):
        if videos.dim() != 5:
            raise ValueError(f"Expected videos [B,T,C,H,W], got {tuple(videos.shape)}")

        videos = videos.to(self.device)
        labels = labels.to(self.device).long()
        used_labels = self._labels_for_loss(videos, labels).long()

        original_01 = denormalize_video(videos.detach(), self.mean, self.std).clamp(0, 1)
        if self.random_start and self.epsilon > 0:
            adv_01 = original_01 + torch.empty_like(original_01).uniform_(-self.epsilon, self.epsilon)
            adv_01 = adv_01.clamp(0, 1)
            adv_videos = normalize_video(adv_01, self.mean, self.std).detach()
        else:
            adv_videos = videos.detach().clone()

        for _ in range(self.steps):
            adv_videos = adv_videos.detach()
            adv_videos.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(adv_videos)
            grad = torch.autograd.grad(
                self._loss(logits, used_labels),
                adv_videos,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

            adv_01 = denormalize_video(adv_videos.detach(), self.mean, self.std)
            adv_01 = adv_01 + self.step_size * grad.sign()
            delta = (adv_01 - original_01).clamp(-self.epsilon, self.epsilon)
            adv_01 = (original_01 + delta).clamp(0, 1)
            adv_videos = normalize_video(adv_01, self.mean, self.std).detach()

        return adv_videos


__all__ = ["ProjectedGradientDescent"]
