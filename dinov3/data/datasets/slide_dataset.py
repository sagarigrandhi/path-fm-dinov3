# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from openslide import OpenSlide
from PIL import Image

from .extended import ExtendedVisionDataset


@dataclass(frozen=True)
class _PatchSpec:
    path: Path
    x: int
    y: int
    level: int


class SlideDataset(ExtendedVisionDataset):
    """
    Minimal dataset that yields on-the-fly pathology patches described in a text spec.

    Each spec line must contain "<absolute_slide_path> <x> <y> <level>" separated by whitespace,
    matching the format used throughout OpenMidnight.
    """

    def __init__(
        self,
        *,
        root: str,
        spec: str | None = None,
        patch_size: int = 224,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.root = Path(root).expanduser()
        self.patch_size = patch_size
        self.spec_path = self._resolve_spec_path(spec)
        self._patch_specs = self._load_specs(self.spec_path)

    def _resolve_spec_path(self, spec: str | None) -> Path:
        default_spec = Path("/data/TCGA/sample_dataset_30.txt")
        candidate = Path(spec) if spec else default_spec
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Unable to locate pathology spec file: {candidate}")
        return candidate

    def _load_specs(self, spec_path: Path) -> List[_PatchSpec]:
        specs: List[_PatchSpec] = []
        with spec_path.open("r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid spec line (expected 4 tokens): {line}")
                slide_path = Path(parts[0])
                if not slide_path.is_absolute():
                    slide_path = (self.root / slide_path).resolve()
                specs.append(
                    _PatchSpec(
                        path=slide_path,
                        x=int(parts[1]),
                        y=int(parts[2]),
                        level=int(parts[3]),
                    )
                )

        if not specs:
            raise ValueError(f"No patch specs found in {spec_path}")
        return specs

    def __getitem__(self, index: int):
        spec = self._patch_specs[index]
        slide = OpenSlide(str(spec.path))
        region = slide.read_region(
            (spec.x, spec.y),
            level=spec.level,
            size=(self.patch_size, self.patch_size),
        ).convert("RGB")
        slide.close()

        image: Image.Image = region
        target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._patch_specs)
