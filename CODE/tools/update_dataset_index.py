#!/usr/bin/env python3
"""扫描 data/LH-data/transfer-dynamic 下的数据集，生成网页查看器的下拉框索引。

每个数据集目录内查找 points3d.ply / points3D.ply 作为默认初始点云。
新增数据集后重新运行本脚本即可：

    python tools/update_dataset_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_DYNAMIC = REPO_ROOT / "data" / "LH-data" / "transfer-dynamic"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "datasets.json"
GT_LIGHTS_REL = "lhdata/danamic/only_clothV3/lights.json"


def main() -> None:
    if not TRANSFER_DYNAMIC.is_dir():
        raise SystemExit(f"数据集目录不存在: {TRANSFER_DYNAMIC}")

    datasets = []
    for entry in sorted(TRANSFER_DYNAMIC.iterdir()):
        if not entry.is_dir():
            continue
        ply = None
        for candidate in ("points3d.ply", "points3D.ply"):
            if (entry / candidate).exists():
                ply = f"lhdata/transfer-dynamic/{entry.name}/{candidate}"
                break
        datasets.append({
            "name": entry.name,
            "ply": ply,
            "has_transforms": (entry / "transforms_train.json").exists(),
        })

    index = {
        "gt_lights": GT_LIGHTS_REL,
        "transfer_dynamic_root": "lhdata/transfer-dynamic",
        "datasets": datasets,
    }
    OUTPUT.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {OUTPUT}（{len(datasets)} 个数据集）")
    for dataset in datasets:
        print(f"  - {dataset['name']}: {dataset['ply'] or '缺少 points3d.ply'}")


if __name__ == "__main__":
    main()
