#!/usr/bin/env python3
"""静态DL冒烟入口：逐步断言几何为零并审计d_color反向梯度。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

from scripts import train_static_dl as training_entry


class AuditedStaticDLTrainer(training_entry.StaticDLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_path = Path(self.args.model_path) / "smoke_gradient_audit.jsonl"
        self.summary_path = Path(self.args.model_path) / "smoke_gradient_audit.json"
        if self.audit_path.exists() or self.summary_path.exists():
            raise FileExistsError("冒烟梯度审计文件已存在，拒绝覆盖。")
        self.audit_records = []
        self.deform_hook = self.deform.deform.register_forward_hook(self._forward_hook)

    def _assert_zero_geometry(self, output, label):
        for key in ("d_xyz", "d_rotation", "d_scaling"):
            value = output.get(key)
            if not torch.is_tensor(value):
                raise AssertionError(f"{label}: {key}不是Tensor，无法验证零形变。")
            if not torch.isfinite(value).all() or torch.count_nonzero(value).item() != 0:
                raise AssertionError(f"{label}: {key}不是逐元素严格零。")

    def _forward_hook(self, module, inputs, output):
        self._assert_zero_geometry(output, f"train_forward_iter_{self.iteration}")
        for name in ("gaussian_warp", "gaussian_rotation", "gaussian_scaling"):
            layer = getattr(module, name)
            if layer.weight.requires_grad or layer.bias.requires_grad:
                raise AssertionError(f"{name}意外参与梯度更新。")
            if torch.count_nonzero(layer.weight).item() or torch.count_nonzero(layer.bias).item():
                raise AssertionError(f"{name}权重或偏置发生变化。")
        color = output.get("d_color")
        if not torch.is_tensor(color):
            raise AssertionError("d_color没有生成Tensor。")
        if color.requires_grad:
            iteration = int(self.iteration)

            def save_gradient(gradient):
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"iteration {iteration}: d_color梯度缺失或非有限。")
                norm = float(gradient.norm().detach())
                max_abs = float(gradient.abs().max().detach())
                record = {"iteration": iteration, "d_color_grad_norm": norm,
                          "d_color_grad_max_abs": max_abs,
                          "d_color_grad_nonzero": bool(torch.count_nonzero(gradient).item())}
                self.audit_records.append(record)
                with self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if norm <= 0 or max_abs <= 0:
                    raise FloatingPointError(f"iteration {iteration}: d_color梯度全零。")
                return gradient

            color.register_hook(save_gradient)

    @torch.no_grad()
    def _verify_all_times(self):
        network = self.deform.deform
        xyz = self.gaussians.get_xyz[:8].detach()
        if xyz.shape[0] == 0:
            raise AssertionError("没有可用于零形变复核的Gaussian。")
        feature = torch.ones((len(xyz), self.dataset.hyper_dim), device=xyz.device)
        camera_center = torch.zeros_like(xyz)
        for value in torch.linspace(0.0, 1.0, 151, device=xyz.device):
            time_input = torch.full((len(xyz), 1), value.item(), device=xyz.device)
            output = network(xyz, time_input, iteration=int(value.item()*150000),
                             feature=feature, camera_center=camera_center)
            self._assert_zero_geometry(output, f"all_time_{value.item():.6f}")
        return 151

    def train(self, iters=5000):
        try:
            super().train(iters)
            expected_steps = int(iters)
            if len(self.audit_records) != expected_steps:
                raise AssertionError(f"d_color梯度记录数不符：{len(self.audit_records)} != {expected_steps}")
            all_times = self._verify_all_times()
            for name in ("gaussian_warp", "gaussian_rotation", "gaussian_scaling"):
                layer = getattr(self.deform.deform, name)
                if layer.weight.count_nonzero().item() or layer.bias.count_nonzero().item():
                    raise AssertionError(f"训练结束时{name}非零。")
            payload = {
                "status": "PASS",
                "training_steps": expected_steps,
                "gradient_records": len(self.audit_records),
                "all_step_d_color_grad_nonzero": all(r["d_color_grad_norm"] > 0 for r in self.audit_records),
                "geometry_output_zero_at_each_forward": True,
                "geometry_heads_zero_and_frozen_after_training": True,
                "arbitrary_time_samples_checked": all_times,
                "minimum_d_color_gradient_norm": min(r["d_color_grad_norm"] for r in self.audit_records),
                "maximum_d_color_gradient_norm": max(r["d_color_grad_norm"] for r in self.audit_records),
            }
            if not payload["all_step_d_color_grad_nonzero"]:
                raise AssertionError("至少一个训练步的d_color梯度为零。")
            self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
            print("[静态DL冒烟审计] " + json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception as error:
            payload = {"status": "FAILED", "error": repr(error),
                       "gradient_records": len(self.audit_records), "records": self.audit_records}
            self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
            raise
        finally:
            self.deform_hook.remove()


def main():
    training_entry.StaticDLTrainer = AuditedStaticDLTrainer
    training_entry.main()


if __name__ == "__main__":
    main()
