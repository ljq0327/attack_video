from .bim import BasicIterativeMethod
from .pgd import ProjectedGradientDescent
from .temporal_translation import ATTACK_REGISTRY as TT_ATTACK_REGISTRY
from .temporal_translation import TemporalTranslation, TemporalTranslationTI


ATTACK_REGISTRY = {
    **TT_ATTACK_REGISTRY,
    "pgd": ProjectedGradientDescent,
    "PGD": ProjectedGradientDescent,
    "projected_gradient_descent": ProjectedGradientDescent,
    "bim": BasicIterativeMethod,
    "BIM": BasicIterativeMethod,
    "basic_iterative_method": BasicIterativeMethod,
}


def build_attack(name, model, **kwargs):
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unsupported attack_method: {name}. Available: {sorted(ATTACK_REGISTRY)}")

    attack_cls = ATTACK_REGISTRY[name]
    if attack_cls is TemporalTranslation:
        kwargs.pop("ti_kernlen", None)
        kwargs.pop("ti_nsig", None)
        kwargs.pop("pgd_step_size", None)
        kwargs.pop("pgd_random_start", None)
    elif attack_cls is TemporalTranslationTI:
        kwargs.pop("pgd_step_size", None)
        kwargs.pop("pgd_random_start", None)
    elif attack_cls in (ProjectedGradientDescent, BasicIterativeMethod):
        for key in [
            "kernlen",
            "momentum",
            "move_type",
            "kernel_mode",
            "delay",
            "shift_batch_size",
            "ti_kernlen",
            "ti_nsig",
        ]:
            kwargs.pop(key, None)

    return attack_cls(model=model, **kwargs)


__all__ = [
    "BasicIterativeMethod",
    "ProjectedGradientDescent",
    "TemporalTranslation",
    "TemporalTranslationTI",
    "build_attack",
]
