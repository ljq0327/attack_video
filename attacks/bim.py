from .pgd import ProjectedGradientDescent


class BasicIterativeMethod(ProjectedGradientDescent):
    """L-infinity BIM attack: iterative sign updates without random start."""

    def __init__(self, *args, **kwargs):
        kwargs["pgd_random_start"] = False
        super().__init__(*args, **kwargs)


__all__ = ["BasicIterativeMethod"]
