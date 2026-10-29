from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
PYTORCHVIDEO_ROOT = THIRD_PARTY_ROOT / "facebookresearch-pytorchvideo-7da857b"
TIMESFORMER_ROOT = THIRD_PARTY_ROOT / "TimeSformer-main"
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
TIMESFORMER_K400_PRETRAINED = PRETRAINED_ROOT / "TimeSformer_divST_8x32_224_K400.pyth"


def add_third_party_paths():
    for path in (TIMESFORMER_ROOT, PYTORCHVIDEO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
