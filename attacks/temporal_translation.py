import random

import torch
import torch.nn.functional as F


DEFAULT_MEAN = (0.45, 0.45, 0.45)
DEFAULT_STD = (0.225, 0.225, 0.225)


def _channel_stats(values, device, dtype):
    return torch.as_tensor(values, device=device, dtype=dtype).view(1, 1, -1, 1, 1)


def normalize_video(video_01, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    mean = _channel_stats(mean, video_01.device, video_01.dtype)
    std = _channel_stats(std, video_01.device, video_01.dtype)
    return (video_01 - mean) / std


def denormalize_video(video, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    mean = _channel_stats(mean, video.device, video.dtype)
    std = _channel_stats(std, video.device, video.dtype)
    return video * std + mean


def normalize_grads(grads, frame_level=True, eps=1e-12):
    if grads.dim() != 5:
        raise ValueError(f"Expected gradients [B,T,C,H,W], got {tuple(grads.shape)}")
    if frame_level:
        denom = grads.abs().mean(dim=(2, 3, 4), keepdim=True)
    else:
        denom = grads.abs().mean(dim=(1, 2, 3, 4), keepdim=True)
    return grads / denom.clamp_min(eps)


def _make_1d_kernel(kernlen, mode, device, dtype):
    if kernlen < 1 or kernlen % 2 != 1:
        raise ValueError(f"kernlen must be a positive odd integer, got {kernlen}")

    if kernlen == 1:
        kernel = torch.ones(1, device=device, dtype=dtype)
    elif mode == "uniform":
        kernel = torch.ones(kernlen, device=device, dtype=dtype)
    elif mode == "linear":
        radius = (kernlen - 1) // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype).abs()
        kernel = radius + 1 - coords
    elif mode == "gaussian":
        radius = (kernlen - 1) // 2
        sigma = max(radius / 3.0, 1e-6)
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unsupported kernel_mode: {mode}")

    return kernel / kernel.sum()


def _make_2d_gaussian_kernel(kernlen, nsig, channels, device, dtype):
    if kernlen < 1 or kernlen % 2 != 1:
        raise ValueError(f"ti_kernlen must be a positive odd integer, got {kernlen}")

    coords = torch.linspace(-nsig, nsig, kernlen, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords ** 2) / 2)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernlen, kernlen).repeat(channels, 1, 1, 1)


class ProjectVideoModel(torch.nn.Module):
    """Adapter: [B,T,C,H,W] -> logits [B,num_classes] for this project."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, videos):
        if videos.dim() != 5:
            raise ValueError(f"Expected videos [B,T,C,H,W], got {tuple(videos.shape)}")

        outputs = self.model(videos.unsqueeze(1))
        if isinstance(outputs, dict):
            if "per_view_logits" in outputs:
                return outputs["per_view_logits"].mean(dim=1)
            if "logits" in outputs:
                return outputs["logits"]
            raise KeyError(f"Unsupported model output keys: {sorted(outputs.keys())}")

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if not torch.is_tensor(outputs):
            raise TypeError(f"Unsupported model output type: {type(outputs)}")
        if outputs.dim() == 3:
            return outputs.mean(dim=1)
        if outputs.dim() == 2:
            return outputs
        raise ValueError(f"Unsupported logits shape: {tuple(outputs.shape)}")


class TemporalTranslationBase:
    def __init__(self, model, epsilon=16/255, steps=10,
        kernlen=15, momentum=False, move_type="adj", kernel_mode="gaussian", delay=1.0, mean=DEFAULT_MEAN, std=DEFAULT_STD,
        shift_batch_size=5, targeted=False, target_map_function=None,
    ):
        self.model = ProjectVideoModel(model)
        self.training = model.training
        self.device = next(model.parameters()).device
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.kernlen = int(kernlen)
        self.momentum = bool(momentum)
        self.move_type = move_type
        self.kernel_mode = kernel_mode
        self.delay = float(delay)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.shift_batch_size = max(1, int(shift_batch_size))
        self.targeted = bool(targeted)
        self.target_map_function = target_map_function

        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if self.move_type not in {"adj", "remote", "random"}:
            raise ValueError(f"Unsupported move_type: {move_type}")

        self.step_size = self.epsilon / self.steps
        self.kernel = _make_1d_kernel(
            self.kernlen,
            self.kernel_mode,
            self.device,
            torch.float32,
        )

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

    def _translation_shifts(self):
        radius = (self.kernlen - 1) // 2
        return list(range(-radius, radius + 1))

    def _actual_shift(self, shift, frames):
        if frames <= 0:
            raise ValueError("videos must contain at least one frame")

        if self.move_type == "adj":
            return shift % frames

        if self.move_type == "remote":
            if shift == 0:
                return 0
            direction = -1 if shift < 0 else 1
            moved = (abs(shift) + (frames // 2 - 1)) % frames
            return (direction * moved) % frames

        if shift == 0:
            return 0
        direction = -1 if shift < 0 else 1
        return (direction * (random.randint(0, 100) % frames)) % frames

    def _roll_time(self, videos, shift):
        return torch.roll(videos, shifts=int(shift), dims=1)

    def _model_logits(self, videos):
        return self.model(videos)

    def _spatial_smooth_grad(self, grad):
        return grad

    def _grad_for_shift_batch(self, adv_videos, labels, shifts):
        batch_size = adv_videos.size(0)
        frames = adv_videos.size(1)
        shifted_videos = []
        actual_shifts = []

        for shift in shifts:
            actual_shift = self._actual_shift(shift, frames)
            shifted_videos.append(self._roll_time(adv_videos, actual_shift))
            actual_shifts.append(actual_shift)

        batch_videos = torch.cat(shifted_videos, dim=0).detach()
        batch_videos.requires_grad_(True)
        batch_labels = labels.repeat(len(shifts))

        self.model.zero_grad(set_to_none=True)
        logits = self._model_logits(batch_videos)
        grad = torch.autograd.grad(
            self._loss(logits, batch_labels),
            batch_videos,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

        restored = []
        for actual_shift, grad_part in zip(actual_shifts, grad.split(batch_size, dim=0)):
            restored.append(self._roll_time(grad_part, -actual_shift))
        return restored

    def _translation_grad(self, adv_videos, labels):
        shifts = self._translation_shifts()
        grads = []

        for start in range(0, len(shifts), self.shift_batch_size):
            shift_batch = shifts[start : start + self.shift_batch_size]
            grads.extend(self._grad_for_shift_batch(adv_videos, labels, shift_batch))

        grad_stack = torch.stack(grads, dim=0)
        weights = self.kernel.to(device=adv_videos.device, dtype=adv_videos.dtype)
        view_shape = (weights.numel(),) + (1,) * (grad_stack.dim() - 1)
        grad = (grad_stack * weights.view(view_shape)).sum(dim=0)
        return self._spatial_smooth_grad(grad)

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
        adv_videos = videos.detach().clone()
        momentum_grad = torch.zeros_like(adv_videos)

        for _ in range(self.steps):
            grad = self._translation_grad(adv_videos, used_labels)
            if self.momentum:
                grad = normalize_grads(grad, frame_level=True)
                grad = grad + momentum_grad * self.delay
                momentum_grad = grad

            adv_01 = denormalize_video(adv_videos.detach(), self.mean, self.std)
            adv_01 = adv_01 + self.step_size * grad.sign()
            delta = (adv_01 - original_01).clamp(-self.epsilon, self.epsilon)
            adv_01 = (original_01 + delta).clamp(0, 1)
            adv_videos = normalize_video(adv_01, self.mean, self.std).detach()

        return adv_videos


class TemporalTranslation(TemporalTranslationBase):
    """Temporal Translation attack, including TT-MI when momentum=True."""


class TemporalTranslationTI(TemporalTranslationBase):
    """Temporal Translation with spatial translation-invariant gradient smoothing."""

    def __init__(self, *args, ti_kernlen=15, ti_nsig=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ti_kernlen = int(ti_kernlen)
        self.ti_nsig = float(ti_nsig)
        self._spatial_kernel = None

    def _spatial_smooth_grad(self, grad):
        batch, frames, channels, height, width = grad.shape
        if self._spatial_kernel is None or self._spatial_kernel.size(0) != channels:
            self._spatial_kernel = _make_2d_gaussian_kernel(
                self.ti_kernlen,
                self.ti_nsig,
                channels,
                grad.device,
                grad.dtype,
            )
        kernel = self._spatial_kernel.to(device=grad.device, dtype=grad.dtype)
        smoothed = F.conv2d(
            grad.reshape(batch * frames, channels, height, width),
            kernel,
            groups=channels,
            padding=self.ti_kernlen // 2,
        )
        smoothed = smoothed.reshape(batch, frames, channels, height, width)
        return normalize_grads(smoothed, frame_level=True)


ATTACK_REGISTRY = {
    "tt": TemporalTranslation,
    "temporaltranslation": TemporalTranslation,
    "temporal_translation": TemporalTranslation,
    "TemporalTranslation": TemporalTranslation,
    "tt_ti": TemporalTranslationTI,
    "tti": TemporalTranslationTI,
    "temporaltranslation_ti": TemporalTranslationTI,
    "temporal_translation_ti": TemporalTranslationTI,
    "TemporalTranslation_TI": TemporalTranslationTI,
}


def build_attack(name, model, **kwargs):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unsupported attack_method: {name}. Available: {sorted(ATTACK_REGISTRY)}")
    attack_cls = ATTACK_REGISTRY[name]
    if attack_cls is TemporalTranslation:
        kwargs.pop("ti_kernlen", None)
        kwargs.pop("ti_nsig", None)
    return attack_cls(model=model, **kwargs)


__all__ = [
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "TemporalTranslation",
    "TemporalTranslationTI",
    "build_attack",
    "denormalize_video",
    "normalize_video",
]
