import torch

from models.model_i3d import I3DClassifier
from models.model_slowfast import SlowFastClassifier
from models.model_timesformer import TimeSformerClassifier
from models.model_x3d import X3DClassifier
from models.model_dvanet import DVANETClassifier
from models.model_tbcnet import TBCNetClassifier
from models.model_ccacnet import CCACNetClassifier
from project_paths import TIMESFORMER_K400_PRETRAINED


LEGACY_STATE_PREFIXES = {
    "intra_view_enhancer.": "intra_view_refiner.",
    "view_center.": "view_weight_estimator.",
    "noise_estimator.": "compensation_direction_predictor.",
}

CCAC_AUXILIARY_PREFIXES = (
    "intra_view_refiner.",
    "view_weight_estimator.",
    "compensation_direction_predictor.",
)


def build_model(model_name, cfg):
    if model_name == "i3d":
        return I3DClassifier(cfg.num_actions, model_name="i3d_r50", pretrained=True)

    if model_name == "slowfast":
        return SlowFastClassifier(cfg.num_actions, model_name="slowfast_r50", pretrained=True)

    if model_name == "timesformer":
        return TimeSformerClassifier(
            num_actions=cfg.num_actions,
            num_frames=16,
            pretrained_model=str(TIMESFORMER_K400_PRETRAINED),
        )

    if model_name == "x3d":
        return X3DClassifier(cfg.num_actions, model_name="x3d_m", pretrained=True)

    if model_name == "dvanet":
        return DVANETClassifier(
            cfg.num_actions,
            num_views=getattr(cfg, "num_views", 3),
            dataset=getattr(cfg, "dataset", None),
        )

    if model_name == "tbcnet":
        return TBCNetClassifier(
            cfg.num_actions,
            num_views=getattr(cfg, "num_views", 3),
            device="cpu",
        )

    if model_name == "ccacnet":
        return CCACNetClassifier(cfg.num_actions)

    raise ValueError(f"Unsupported model_name: {model_name}")


def _extract_checkpoint_state_dict(checkpoint, use_ema=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if use_ema and checkpoint.get("ema_state_dict") is not None:
        return checkpoint["ema_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _normalize_checkpoint_key(key):
    if key.startswith("module."):
        key = key.replace("module.", "", 1)

    for old_prefix, new_prefix in LEGACY_STATE_PREFIXES.items():
        if key.startswith(old_prefix):
            key = key.replace(old_prefix, new_prefix, 1)
            break

    return key


def _normalize_checkpoint_state_dict(state_dict):
    return {
        _normalize_checkpoint_key(key): value
        for key, value in state_dict.items()
    }


def _strip_prefix_if_all_keys_match(state_dict, prefix):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {
            key.replace(prefix, "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def _configure_ccacnet_from_state_dict(target_model, state_dict, strict):
    if not hasattr(target_model, "set_auxiliary_modules_enabled"):
        return strict

    has_auxiliary_weights = any(
        key.startswith(CCAC_AUXILIARY_PREFIXES)
        for key in state_dict
    )
    target_model.set_auxiliary_modules_enabled(has_auxiliary_weights)
    if not has_auxiliary_weights:
        strict = False
    return strict


def load_checkpoint_state_dict(model, checkpoint, model_name=None, use_ema=False, strict=True):
    state_dict = _extract_checkpoint_state_dict(checkpoint, use_ema=use_ema)
    state_dict = _normalize_checkpoint_state_dict(state_dict)

    target_model = model.module if hasattr(model, "module") else model
    if model_name in {"dvanet", "tbcnet"}:
        if model_name == "tbcnet":
            strict = False
        state_dict = _strip_prefix_if_all_keys_match(state_dict, "model.")
        target_model.model.load_state_dict(state_dict, strict=strict)
    elif model_name == "ccacnet":
        if hasattr(target_model, "model"):
            state_dict = _strip_prefix_if_all_keys_match(state_dict, "model.")
            state_dict = _normalize_checkpoint_state_dict(state_dict)
            strict = _configure_ccacnet_from_state_dict(target_model.model, state_dict, strict)
            target_model.model.load_state_dict(state_dict, strict=strict)
        else:
            strict = _configure_ccacnet_from_state_dict(target_model, state_dict, strict)
            target_model.load_state_dict(state_dict, strict=strict)
    else:
        target_model.load_state_dict(state_dict, strict=strict)

    return model


def load_model_checkpoint(model, checkpoint_path, model_name=None, use_ema=False, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return load_checkpoint_state_dict(
        model,
        checkpoint,
        model_name=model_name,
        use_ema=use_ema,
        strict=strict,
    )
