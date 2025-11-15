"""Utility to wrap published backbone weights into a full teacher checkpoint.

The official DINOv3 ViT-H/16 checkpoint from Meta only provides a backbone
state_dict, while our fine-tuning code expects a consolidated "teacher"
checkpoint containing a ModuleDict with backbone + DINO/iBOT heads. This
script loads the torch.hub backbone weights, instantiates fresh heads with the
desired dimensions, and saves the combined state under a top-level "teacher"
key so SSLMetaArch can resume without missing-key errors.
"""

import argparse
import sys
from pathlib import Path

import torch

# Local repo path so torch.hub can load from source
REPO_DIR = "/home/paul/dinov3"


def main():
    parser = argparse.ArgumentParser(
        description="Load DINOv3 vithplus backbone and save a pretraining-style teacher checkpoint."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/dinov3_vith16plus_saved_teacher.pth",
        help="Output path for the saved teacher checkpoint (.pth).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        help="Path or URL to pretrained backbone weights.",
    )
    # Head sizing defaults/overrides (avoids relying on any external YAML)
    parser.add_argument("--dino-prototypes", type=int, default=262144)
    parser.add_argument("--dino-bottleneck-dim", type=int, default=512)
    parser.add_argument("--dino-hidden-dim", type=int, default=8192)
    parser.add_argument("--dino-nlayers", type=int, default=3)
    parser.add_argument("--ibot-prototypes", type=int, default=98304)
    parser.add_argument("--ibot-bottleneck-dim", type=int, default=384)
    parser.add_argument("--ibot-hidden-dim", type=int, default=4096)
    parser.add_argument("--ibot-nlayers", type=int, default=3)
    args = parser.parse_args()

    # Load backbone from local repo with the provided weights without torch.hub to avoid extra deps
    sys.path.insert(0, str(Path(REPO_DIR)))
    from dinov3.hub.backbones import dinov3_vith16plus  # noqa: E402

    backbone = dinov3_vith16plus(pretrained=True, weights=args.weights)
    backbone.eval()

    embed_dim = getattr(backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Loaded backbone has no embed_dim attribute; cannot size heads.")

    from dinov3.layers.dino_head import DINOHead  # noqa: E402

    dino_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.dino_prototypes,
        hidden_dim=args.dino_hidden_dim,
        bottleneck_dim=args.dino_bottleneck_dim,
        nlayers=args.dino_nlayers,
    )
    ibot_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.ibot_prototypes,
        hidden_dim=args.ibot_hidden_dim,
        bottleneck_dim=args.ibot_bottleneck_dim,
        nlayers=args.ibot_nlayers,
    )
    dino_head.init_weights()
    ibot_head.init_weights()

    moduledict = torch.nn.ModuleDict(
        {
            "backbone": backbone,
            "dino_head": dino_head,
            "ibot_head": ibot_head,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_state = {k: v.cpu() for k, v in moduledict.state_dict().items()}
    torch.save({"teacher": teacher_state}, out_path)
    print(f"Saved teacher checkpoint to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
