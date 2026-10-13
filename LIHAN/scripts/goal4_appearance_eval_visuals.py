#!/usr/bin/env python3
"""仅适配appearance物理前向，复用原150帧四类及几何map一致性。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import goal_eval_render,goal_eval_visuals
from goal4_appearance_eval_render import physical_renderer
from goal_eval_protocol import digest
def renderer(run,iteration,directions):
    run.files[str(Path(__file__).resolve())]=digest(__file__)
    return physical_renderer(run,iteration,directions)
def main():
    goal_eval_render.physical_renderer=renderer
    goal_eval_visuals.main()
if __name__=='__main__':main()
