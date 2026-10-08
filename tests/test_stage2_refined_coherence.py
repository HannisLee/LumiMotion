"""真实训练状态复制、显式邻域重建和独立检查点恢复。"""
import copy
import unittest
from argparse import ArgumentParser

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual
from scene.stage2_rgb_prior import initialize_geometry_from_stage2, COHERENCE_STATE_FIELDS
from scripts.loss_stage2 import apply_loss_preset


class RefinedCoherenceTest(unittest.TestCase):
    def setUp(self):
        u, v = torch.meshgrid(torch.arange(4.), torch.arange(4.), indexing='ij')
        self.xyz = torch.stack((u.flatten(), v.flatten(), u.flatten()*0), -1)

    def args(self, **kwargs):
        parser = ArgumentParser(); Stage2OptimizationParams(parser); Stage2PipelineParams(parser)
        args = parser.parse_args([])
        for k, v in kwargs.items(): setattr(args, k, v)
        return args

    def state(self, g, **config):
        return dict(geometry=copy.deepcopy(g.state_dict()), config=config)

    def coherent(self, refined=False, rgb=False):
        return GeometryResidual(self.xyz, self.args(lambda_geometry_coherence=.01,
            geometry_coherence_source_mask='saved_mask.npz', geometry_refined_coherence=refined,
            geometry_initial_stage2_model='real_trained_model',
            lambda_geometry_rgb_normal_prior=2. if rgb else 0., geometry_rgb_normal_prior_path='saved_rgb.npz'))

    def build(self, g):
        q = torch.tensor([1., 0., 0., 0.]).expand(16, -1)
        means, _, scales, opacity = g.prepare(self.xyz, q, torch.ones(16, 2), torch.ones(16, 1))
        g.coherence_source_static.fill_(True)
        g.initialize_coherence(means, q[:, 1:].clone().add(torch.tensor([0., 0., 1.])), scales, opacity,
                               static=g.coherence_source_static)

    def test_old_defaults_and_explicit_zero_keep_state_unchanged(self):
        args = self.args(); self.assertFalse(args.geometry_refined_coherence)
        old = GeometryResidual(self.xyz, args)
        args.render_mode='photometric_lambertian'; args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_refined_coherent'
        args.lambda_geometry_coherence=0.
        apply_loss_preset(args, ['--lambda_geometry_coherence', '0'])
        zero = GeometryResidual(self.xyz, args)
        self.assertTrue(args.geometry_refined_coherence)
        self.assertFalse(zero.coherence_enabled)
        # 此preset会启用细分和scale；显式关闭后比较旧零残差状态。
        args.geometry_subdivide=False; args.geometry_optimize_scale=False; args.geometry_tonemap='srgb'
        zero = GeometryResidual(self.xyz, args)
        self.assertEqual(old.state_dict().keys(), zero.state_dict().keys())
        for k, v in old.state_dict().items(): torch.testing.assert_close(v, zero.state_dict()[k], rtol=0, atol=0)

    def test_active_refinement_requires_real_source_and_training_mask(self):
        for source, mask in [('', 'mask.npz'), ('model', '')]:
            with self.assertRaises(ValueError):
                GeometryResidual(self.xyz, self.args(lambda_geometry_coherence=.01,
                    geometry_refined_coherence=True, geometry_initial_stage2_model=source,
                    geometry_coherence_source_mask=mask))

    def test_existing_rgb_cache_and_learned_state_copy_exactly(self):
        args = self.args(lambda_geometry_rgb_normal_prior=2., geometry_rgb_normal_prior_path='unused.npz')
        source=GeometryResidual(self.xyz,args)
        with torch.no_grad():
            source.translation[:,2]=.01; source.axis_angle[:,0]=.02
            source.rgb_normal_targets[:,2]=1.; source.rgb_normal_valid[:8]=True
        state=self.state(source,lambda_geometry_rgb_normal_prior=2.)
        target=GeometryResidual(self.xyz,args)
        result=initialize_geometry_from_stage2(target,state)
        self.assertFalse(result['refined_coherence_graph_rebuilt'])
        for k,v in state['geometry'].items(): torch.testing.assert_close(target.state_dict()[k],v,rtol=0,atol=0)

    def test_new_rgb_cache_is_only_permitted_missing_buffer(self):
        source=GeometryResidual(self.xyz,self.args())
        target=GeometryResidual(self.xyz,self.args(lambda_geometry_rgb_normal_prior=2.,geometry_rgb_normal_prior_path='new.npz'))
        initialize_geometry_from_stage2(target,self.state(source))
        self.assertFalse(target.rgb_normal_valid.any())
        for bad in [dict(self.state(source)['geometry'],unknown=torch.ones(1)),
                    {k:v for k,v in source.state_dict().items() if k!='axis_angle'}]:
            with self.assertRaises(ValueError): initialize_geometry_from_stage2(target,dict(geometry=bad,config={}))

    def test_corrupt_or_incomplete_rgb_sources_rejected(self):
        source=GeometryResidual(self.xyz,self.args(lambda_geometry_rgb_normal_prior=2.,geometry_rgb_normal_prior_path='source.npz'))
        source.rgb_normal_targets[:,2]=1.; source.rgb_normal_valid[:8]=True
        state=self.state(source,lambda_geometry_rgb_normal_prior=2.)
        cases=[]
        a=copy.deepcopy(state); del a['geometry']['rgb_normal_valid']; cases.append(a)
        a=copy.deepcopy(state); del a['geometry']['rgb_normal_valid']; del a['geometry']['rgb_normal_targets']; cases.append(a)
        a=copy.deepcopy(state); a['geometry']['rgb_normal_targets']*=2; cases.append(a)
        a=copy.deepcopy(state); a['geometry']['rgb_normal_valid'].fill_(False); cases.append(a)
        for case in cases:
            with self.assertRaises(ValueError): initialize_geometry_from_stage2(source,case)

    def test_refined_graph_rebuilds_from_loaded_surface_and_survives_reload(self):
        source=self.coherent(); self.build(source)
        with torch.no_grad(): source.translation[:,0]=self.xyz[:,0]*.2
        state=self.state(source,lambda_geometry_coherence=.01,geometry_coherence_source_mask='source.npz')
        target=self.coherent(refined=True)
        result=initialize_geometry_from_stage2(target,state,refined_coherence=True)
        self.assertEqual(set(result['recomputed_buffer_names']),COHERENCE_STATE_FIELDS)
        self.assertFalse(target.coherence_initialized)
        for k in ['translation','axis_angle','scene_extent']:
            torch.testing.assert_close(target.state_dict()[k],state['geometry'][k],rtol=0,atol=0)
        self.build(target)
        active=(target.coherence_weights>0)&(source.coherence_weights>0)
        self.assertGreater(float((target.coherence_lengths[active]-source.coherence_lengths[active]).abs().max()),.01)
        self.assertTrue(torch.equal(source.coherence_lengths,state['geometry']['coherence_lengths']))
        clone=self.coherent(refined=True); clone.load_state_dict(target.state_dict())
        before=copy.deepcopy(clone.state_dict())
        self.build(clone)
        for k,v in before.items(): torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)

    def test_default_graph_copy_and_missing_graph_rejection(self):
        source=self.coherent(); self.build(source)
        state=self.state(source,lambda_geometry_coherence=.01,geometry_coherence_source_mask='source.npz')
        target=self.coherent(); initialize_geometry_from_stage2(target,state)
        for k,v in source.state_dict().items(): torch.testing.assert_close(v,target.state_dict()[k],rtol=0,atol=0)
        bad=copy.deepcopy(state); del bad['geometry']['coherence_lengths']
        with self.assertRaises(ValueError): initialize_geometry_from_stage2(target,bad,refined_coherence=True)
        fresh=self.state(GeometryResidual(self.xyz,self.args()))
        with self.assertRaises(ValueError): initialize_geometry_from_stage2(target,fresh)


if __name__=='__main__': unittest.main()
