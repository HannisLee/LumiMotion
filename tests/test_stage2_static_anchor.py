"""静态来源固定时间/相机输入，旧模式输入与缓存行为保持。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import patch

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_source_deformation import find_static_anchor_camera
from scripts.loss_stage2 import apply_loss_preset
from scripts.stage2_runtime import FrozenScene


class StaticAnchorTests(unittest.TestCase):
    def cameras(self):
        return [SimpleNamespace(image_name_train_light=str(i+1).zfill(4), fid=torch.tensor([float(i)]),
                                camera_center=torch.tensor([float(i),0.,1.])) for i in range(4)]

    def context(self, anchor):
        calls = []
        def step(xyz, time, **kwargs):
            calls.append((time.clone(), kwargs['camera_center'].clone()))
            return dict(d_xyz=xyz+time, d_rotation=None, d_scaling=None, d_opacity=None)
        context = FrozenScene.__new__(FrozenScene)
        context.pc = SimpleNamespace(get_xyz=torch.zeros(3,3), get_binary_feature=lambda:torch.zeros(3,1))
        context.deform = SimpleNamespace(step=step)
        context.iteration = 35000
        context.static_anchor_camera = anchor
        context.static_deformation_cache = None
        return context, calls

    def test_default_and_preset_are_explicit(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser)
        args=parser.parse_args([])
        self.assertEqual(args.geometry_source_deformation_mode,'dynamic')
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_static_anchor';apply_loss_preset(args,[])
        self.assertEqual(args.geometry_source_deformation_mode,'anchor_camera')
        args.geometry_source_deformation_mode='dynamic'
        apply_loss_preset(args,['--geometry_source_deformation_mode','dynamic'])
        self.assertEqual(args.geometry_source_deformation_mode,'dynamic')

    def test_anchor_must_be_optimization_camera(self):
        cameras=self.cameras();args=SimpleNamespace(geometry_source_deformation_mode='anchor_camera',geometry_static_anchor_key='0002',geometry_validation_stride=2)
        self.assertIs(find_static_anchor_camera(args,cameras,lambda x:x),cameras[1])
        for bad in ['0001','0003','9999']:
            args.geometry_static_anchor_key=bad
            with self.assertRaises(ValueError):find_static_anchor_camera(args,cameras,lambda x:x)

    def test_unknown_mode_rejected_and_legacy_arguments_supported(self):
        cameras=self.cameras()
        self.assertIsNone(find_static_anchor_camera(SimpleNamespace(),cameras,lambda x:x))
        with self.assertRaises(ValueError):find_static_anchor_camera(SimpleNamespace(geometry_source_deformation_mode='bad'),cameras,lambda x:x)

    def test_static_uses_anchor_time_and_center_and_caches_without_mutation(self):
        cameras=self.cameras();context,calls=self.context(cameras[1])
        with patch.object(torch.Tensor,'to',lambda x,*args,**kw:x):
            first=context.deformation(cameras[2]);again=context.deformation(cameras[3],time_override=9.)
        self.assertIs(first,again);self.assertEqual(len(calls),1)
        torch.testing.assert_close(calls[0][0],torch.full((3,1),1.),rtol=0,atol=0)
        torch.testing.assert_close(calls[0][1],cameras[1].camera_center,rtol=0,atol=0)

    def test_dynamic_uses_each_camera_and_does_not_cache(self):
        cameras=self.cameras();context,calls=self.context(None)
        with patch.object(torch.Tensor,'to',lambda x,*args,**kw:x):
            context.deformation(cameras[1]);context.deformation(cameras[3])
        self.assertEqual(len(calls),2);self.assertIsNone(context.static_deformation_cache)
        for record, camera in zip(calls,[cameras[1],cameras[3]]):
            torch.testing.assert_close(record[0],camera.fid.unsqueeze(0).expand(3,-1),rtol=0,atol=0)
            torch.testing.assert_close(record[1],camera.camera_center,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
