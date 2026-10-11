#!/usr/bin/env python3
"""复用原完整150四类导出，在独立进程仅替换真实RGB适配函数为G2 ambient。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_eval_render
from goal2_eval_render import physical_renderer as ambient_renderer
import goal_eval_visuals


def recorded_renderer(run, iteration, directions):
    from goal_eval_protocol import digest
    run.files[str(Path(__file__).resolve())] = digest(Path(__file__))
    return ambient_renderer(run, iteration, directions)


def main():
    # 只当前进程依赖注入，磁盘上的旧helper/原GeometryRun/主指标未修改。
    goal_eval_render.physical_renderer = recorded_renderer
    goal_eval_visuals.main()


if __name__ == '__main__':
    main()
