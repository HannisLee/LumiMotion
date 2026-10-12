#!/usr/bin/env python3
"""足部恢复新入口模型完整150四类，显式实际a0物理适配。"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import goal_eval_render
import goal_eval_visuals
from goal3_foot_eval_render import physical_renderer


def recorded_renderer(run,iteration,directions):
    from goal_eval_protocol import digest
    run.files[str(Path(__file__).resolve())]=digest(Path(__file__))
    return physical_renderer(run,iteration,directions)


def main():
    goal_eval_render.physical_renderer=recorded_renderer
    goal_eval_visuals.main()


if __name__=='__main__':
    main()
