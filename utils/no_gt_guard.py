"""为无GT预处理/估光安装进程级外部数据读取护栏。"""
import os
from pathlib import Path
import sys


def install_no_gt_guard(allowed):
    roots=[Path(p).resolve() for p in allowed]
    def guard(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(values[0])).resolve()
        if any(path==root or (root.is_dir() and path.is_relative_to(root)) for root in roots):
            return
        if path.suffix.lower() in {'.ply','.pth','.pt','.exr','.blend','.npz','.png','.jpg','.jpeg'} or path.name in {'lights.json','object_pose.json','bone_pose.json'}:
            raise PermissionError('无GT流程禁止读取未登记数据/模型：'+str(path))
    sys.addaudithook(guard)
