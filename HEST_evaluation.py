import argparse
from pathlib import Path

from hest.bench import benchmark
from omegaconf import OmegaConf
import torch
from torchvision import transforms

from dinov3.configs import get_default_config
from dinov3.models import build_model_for_eval

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(
    "/data/path-fm-dinov3/output_vith16plus_8gpu/eval/training_125000/teacher_checkpoint.pth"
)
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "dinov3" / "configs" / "train" / "vith16plus_8gpus.yaml"
DEFAULT_HEST_CONFIG = REPO_ROOT / "HEST" / "bench_config" / "bench_config.yaml"

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _merge_config(train_config_path: Path):
    base_cfg = get_default_config()
    train_cfg = OmegaConf.load(train_config_path)
    return OmegaConf.merge(base_cfg, train_cfg)


def _load_vith16plus_model(checkpoint_path: Path, train_config_path: Path):
    cfg = _merge_config(train_config_path)
    model = build_model_for_eval(cfg, str(checkpoint_path))
    model.eval()
    model.requires_grad_(False)
    return model


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run HEST benchmark with ViT-H+/16 DINOv3 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the finetuned teacher checkpoint (.pth).",
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_TRAIN_CONFIG),
        help="Training config that defines the ViT-H+/16 architecture.",
    )
    parser.add_argument(
        "--hest-config",
        default=str(DEFAULT_HEST_CONFIG),
        help="HEST benchmark YAML config file.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    config_path = Path(args.config_file).expanduser()
    hest_config_path = Path(args.hest_config).expanduser()

    for path, label in (
        (checkpoint_path, "checkpoint"),
        (config_path, "config file"),
        (hest_config_path, "HEST config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} at: {path}")

    if not torch.cuda.is_available():
        raise RuntimeError("ViT-H+/16 DINOv3 evaluation requires a CUDA device.")

    print(f"Loading ViT-H+/16 DINOv3 backbone from {checkpoint_path}")
    model = _load_vith16plus_model(checkpoint_path, config_path)

    model_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    with torch.inference_mode():
        benchmark(
            model,
            model_transforms,
            torch.float32,
            config=str(hest_config_path),
        )


if __name__ == "__main__":
    main()

# USAGE:
# python HEST_evaluation.py --checkpoint /data/path-fm-dinov3/output_vith16plus_8gpu/eval/training_125000/teacher_checkpoint.pth