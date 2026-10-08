"""源预测缓存的恢复精度与训练划分隔离；不使用GT标签。"""
import unittest
from argparse import Namespace

import torch
from scripts.stage2_runtime import restore_source_prediction_caches


class SourceCacheTest(unittest.TestCase):
    def test_prediction_cache_restores_exact_dtype_and_fields(self):
        cameras={'0002':Namespace(image_height=2,image_width=3)}
        reference={'0002':{name:torch.rand(1,2,3) for name in ['depth','alpha','shading']}}
        visibility={'0002':torch.rand(1,2,3).half()}
        a,b=restore_source_prediction_caches(reference,visibility,cameras,True)
        for name in reference['0002']:
            torch.testing.assert_close(a['0002'][name],reference['0002'][name],rtol=0,atol=0)
        torch.testing.assert_close(b['0002'],visibility['0002'],rtol=0,atol=0)
        for bad in [dict(reference, **{'0001':reference['0002']}),
                    {'0002':dict(reference['0002'],gt_normal=torch.zeros(3,2,3))},
                    {'0002':dict(reference['0002'],depth=torch.zeros(1,2,4))},
                    {'0002':dict(reference['0002'],depth=torch.zeros(1,2,3).half())},
                    {'0002':dict(reference['0002'],depth=torch.full((1,2,3),float('nan')))}]:
            with self.assertRaises(ValueError):
                restore_source_prediction_caches(bad,visibility,cameras,True)
        with self.assertRaises(ValueError):
            restore_source_prediction_caches(reference,{'0008':visibility['0002']},cameras,True)


if __name__=='__main__':unittest.main()
