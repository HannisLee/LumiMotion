"""几何残差的旋转梯度、信赖域和 preset 兼容性。"""
import math
import unittest
from argparse import ArgumentParser
import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual, geometry_schedule, apply_filmic_lut, normal_depth_penalty
from scene.stage2_photometric import PointLightTable, AreaLightTable, area_irradiance, make_material, linear_to_srgb
from scripts.loss_stage2 import apply_loss_preset, resolve_preset


class GeometryTest(unittest.TestCase):
    def args(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser)
        return parser.parse_args([])

    def test_identity_and_zero_rotation_gradient(self):
        args=self.args();xyz=torch.randn(20,3);g=GeometryResidual(xyz,args)
        q=torch.tensor([1.,0.,0.,0.]).expand(20,-1)
        p,r=g.apply(xyz,q)
        torch.testing.assert_close(p,xyz);torch.testing.assert_close(r,q)
        (p.sum()+r[:,1].sum()).backward()
        self.assertTrue(torch.isfinite(g.axis_angle.grad).all())
        torch.testing.assert_close(g.axis_angle.grad[:,0],torch.full((20,),.5))

    def test_rotation_composition_projection_and_restore(self):
        args=self.args();xyz=torch.randn(20,3);g=GeometryResidual(xyz,args)
        with torch.no_grad():g.translation.fill_(1);g.axis_angle[:,0]=math.pi/2
        _,r=g.apply(xyz,torch.tensor([1.,0.,0.,0.]).expand(20,-1))
        torch.testing.assert_close(r[:,0],torch.full((20,),math.sqrt(.5)))
        g.project();d=g.diagnostics()
        self.assertLessEqual(d['xyz_max_fraction'],args.geometry_max_displacement+1e-8)
        self.assertLessEqual(d['rotation_max_degrees'],args.geometry_max_angle+1e-5)
        other=GeometryResidual(xyz,args);other.load_state_dict(g.state_dict())
        torch.testing.assert_close(other.apply(xyz,r)[0],g.apply(xyz,r)[0])

    def test_preset_and_schedule(self):
        args=self.args();self.assertEqual(resolve_preset(args,'original_ir'),'irgs_baseline')
        args.loss_preset='lambertian_geometry';args.render_mode='photometric_lambertian'
        args.photometric_light_mode='gt_directional';args.lambda_dssim=.4
        apply_loss_preset(args,['--lambda_dssim','.4'])
        self.assertEqual(args.lambda_dssim,.4)
        self.assertEqual(geometry_schedule(0,2000,args),0)
        self.assertAlmostEqual(geometry_schedule(20000,2000,args),.1)
        with self.assertRaises(ValueError):resolve_preset(args,'original_ir')

    def test_scale_default_identity_and_preset_restore(self):
        args=self.args();xyz=torch.randn(20,3);scales=torch.rand(20,2)
        legacy=GeometryResidual(xyz,args)
        self.assertIs(legacy.apply_scales(scales),scales)
        self.assertNotIn('log_scale',legacy.state_dict())
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_scale';apply_loss_preset(args,[])
        geo=GeometryResidual(xyz,args)
        torch.testing.assert_close(geo.apply_scales(scales),scales,rtol=0,atol=0)
        geo.apply_scales(scales).sum().backward()
        torch.testing.assert_close(geo.log_scale.grad,scales)
        with torch.no_grad():geo.log_scale[:,0]=.1
        restored=GeometryResidual(xyz,args);restored.load_state_dict(geo.state_dict())
        torch.testing.assert_close(restored.apply_scales(scales),geo.apply_scales(scales),rtol=0,atol=0)

    def test_scale_trust_region_clears_clipped_adam_moments(self):
        args=self.args();args.geometry_optimize_scale=True;geo=GeometryResidual(torch.randn(20,3),args)
        optimizer=torch.optim.Adam([geo.log_scale],lr=.01)
        geo.log_scale.sum().backward();optimizer.step()
        with torch.no_grad():geo.log_scale[0,0]=10;geo.log_scale[0,1]=-10
        geo.project(optimizer)
        torch.testing.assert_close(geo.apply_scales(torch.ones(20,2))[0],torch.tensor([1.25,.8]))
        self.assertEqual(float(optimizer.state[geo.log_scale]['exp_avg'][0].abs().sum()),0)
        self.assertGreater(float(optimizer.state[geo.log_scale]['exp_avg'][1].abs().sum()),0)
        args.geometry_max_scale_ratio=1.
        with self.assertRaises(ValueError):GeometryResidual(torch.randn(20,3),args)

    def test_normal_margin_retains_default_and_only_penalizes_excess(self):
        cosine=torch.tensor([1.,math.cos(math.radians(2)),math.cos(math.radians(6))],requires_grad=True)
        torch.testing.assert_close(normal_depth_penalty(cosine),1-cosine,rtol=0,atol=0)
        penalty=normal_depth_penalty(cosine,3.)
        self.assertEqual(float(penalty[:2].sum()),0.)
        self.assertGreater(float(penalty[2]),0.)
        penalty.sum().backward();torch.testing.assert_close(cosine.grad,torch.tensor([0.,0.,-1.]))
        with self.assertRaises(ValueError):normal_depth_penalty(cosine,90.)
        args=self.args();args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_margin';apply_loss_preset(args,[])
        self.assertEqual(args.geometry_normal_margin,3.)
        args.geometry_normal_margin=5.;apply_loss_preset(args,['--geometry_normal_margin','5'])
        self.assertEqual(args.geometry_normal_margin,5.)

    def test_subdivision_preserves_parent_moments_and_linear_opacity_integral(self):
        args=self.args();args.geometry_subdivide=True;xyz=torch.randn(3,3)
        geo=GeometryResidual(xyz,args);q=torch.tensor([1.,0.,0.,0.]).expand(3,-1)
        scales=torch.tensor([[.3,.4],[.1,.2],[.5,.6]]);opacity=torch.tensor([[.9],[.8],[.7]])
        means,rotations,sizes,alphas=geo.prepare(xyz,q,scales,opacity)
        self.assertEqual(means.shape,(12,3))
        torch.testing.assert_close(means.reshape(3,4,3).mean(1),xyz)
        offset=(means-xyz.repeat_interleave(4,0)).reshape(3,4,3)
        covariance=(offset[:,:,:2].square()+sizes.reshape(3,4,2).square()).mean(1)
        torch.testing.assert_close(covariance,scales.square())
        mass=(alphas[:,0]*sizes.prod(-1)).reshape(3,4).sum(1)
        torch.testing.assert_close(mass,opacity[:,0]*scales.prod(-1))
        torch.testing.assert_close(rotations,q.repeat_interleave(4,0))

    def test_coherence_respects_materials_static_mask_and_saved_graph(self):
        args=self.args();args.lambda_geometry_coherence=.05
        y,x=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
        plane=torch.stack((x.flatten(),y.flatten(),torch.zeros(16)),-1).float()*.1
        xyz=torch.cat((plane,plane+torch.tensor([0.,0.,.02])))
        normals=torch.tensor([0.,0.,1.]).expand(32,-1)
        geo=GeometryResidual(xyz,args);labels=torch.arange(2).repeat_interleave(16)
        static=torch.ones(32,dtype=torch.bool);static[0]=False
        geo.initialize_coherence(xyz,normals,torch.full((32,2),.1),torch.ones(32,1),labels,static)
        valid=geo.coherence_weights>0;neighbors=geo.coherence_neighbors
        self.assertTrue(valid.any());self.assertFalse(valid[0].any())
        self.assertFalse((neighbors[valid]==0).any())
        self.assertTrue((labels[:,None].expand_as(neighbors)[valid]==labels[neighbors[valid]]).all())
        saved={k:v.clone() for k,v in geo.state_dict().items()}
        clone=GeometryResidual(xyz,args);clone.load_state_dict(saved)
        clone.initialize_coherence(xyz+torch.randn_like(xyz),normals,torch.ones(32,2),torch.ones(32,1))
        for k,v in clone.state_dict().items():torch.testing.assert_close(v,saved[k],rtol=0,atol=0)

    def test_coherence_preserves_curvature_and_drives_geometry_and_normals(self):
        args=self.args();args.lambda_geometry_coherence=.05
        theta=torch.arange(40)*2*math.pi/40
        normals=torch.stack((theta.cos(),theta.sin(),torch.zeros(40)),-1)
        xyz=normals.clone();geo=GeometryResidual(xyz,args)
        geo.initialize_coherence(xyz,normals,torch.full((40,2),.3),torch.ones(40,1))
        self.assertLess(float(geo.coherence_loss(xyz,normals)),1e-10)
        moved=xyz.clone();moved[0,0]+=.03;moved.requires_grad_(True)
        learned=normals.clone().requires_grad_(True)
        loss=geo.coherence_loss(moved,learned);self.assertGreater(float(loss),0.)
        loss.backward()
        for grad in [moved.grad,learned.grad]:
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0.)
        self.assertGreater(float(moved.grad[0,0]),0.)

    def test_coherence_is_optional_and_explicit_disable_preserves_legacy_state(self):
        args=self.args();xyz=torch.randn(20,3);legacy=GeometryResidual(xyz,args)
        self.assertFalse(legacy.coherence_enabled)
        self.assertFalse(any(k.startswith('coherence_') for k in legacy.state_dict()))
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_coherent';apply_loss_preset(args,[])
        self.assertEqual(args.lambda_geometry_coherence,.05)
        args.lambda_geometry_coherence=0.;apply_loss_preset(args,['--lambda_geometry_coherence','0'])
        self.assertFalse(GeometryResidual(xyz,args).coherence_enabled)

    def test_subdivision_rotation_mapping_gradients_and_restore(self):
        args=self.args();args.geometry_subdivide=True;args.geometry_optimize_scale=True
        xyz=torch.randn(3,3);geo=GeometryResidual(xyz,args)
        q=torch.tensor([math.sqrt(.5),0.,0.,math.sqrt(.5)]).expand(3,-1)
        scales=torch.tensor([[.3,.4],[.1,.2],[.5,.6]]);opacity=torch.ones(3,1)
        means,rotations,sizes,alphas=geo.prepare(xyz,q,scales,opacity)
        local=geo.child_offsets*scales.repeat_interleave(4,0)
        expected=torch.stack((-local[:,1],local[:,0],torch.zeros(12)),-1)
        torch.testing.assert_close(means-xyz.repeat_interleave(4,0),expected,atol=3e-7,rtol=1e-5)
        colors=torch.randn(3,3,requires_grad=True);lifted=geo.expand_values(colors)
        (means.sum()+rotations[:,1].sum()+sizes.sum()+lifted.sum()).backward()
        torch.testing.assert_close(colors.grad,torch.full((3,3),4.))
        for p in geo.parameters():self.assertTrue(torch.isfinite(p.grad).all())
        clone=GeometryResidual(xyz,args);clone.load_state_dict(geo.state_dict())
        for a,b in zip(geo.prepare(xyz,q,scales,opacity),clone.prepare(xyz,q,scales,opacity)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):geo.apply(xyz,q)
        with self.assertRaises(ValueError):geo.prepare(xyz[:2],q[:2],scales[:2],opacity[:2])

    def test_refine_preset_is_optional_and_legacy_state_unchanged(self):
        args=self.args();xyz=torch.randn(3,3);legacy=GeometryResidual(xyz,args)
        self.assertIsNone(legacy.parent_index)
        self.assertNotIn('parent_index',legacy.state_dict())
        self.assertNotIn('child_offsets',legacy.state_dict())
        self.assertIs(legacy.expand_values(xyz),xyz)
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_refine';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_subdivide);self.assertTrue(args.geometry_optimize_scale)
        self.assertEqual(args.geometry_normal_margin,3.)

    def test_dense_subdivision_moments_count_gradient_and_restore(self):
        args=self.args();args.geometry_subdivide=True;args.geometry_subdivision_side=4
        xyz=torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,2.,2.]])
        geo=GeometryResidual(xyz,args);q=torch.tensor([1.,0.,0.,0.]).expand(3,-1)
        scales=torch.tensor([[.3,.4],[.1,.2],[.5,.6]]);opacity=torch.tensor([[.9],[.8],[.7]])
        means,rotations,sizes,alphas=geo.prepare(xyz,q,scales,opacity)
        self.assertEqual(means.shape,(48,3))
        torch.testing.assert_close(means.reshape(3,16,3).mean(1),xyz,rtol=1e-5,atol=2e-7)
        offset=(means-xyz.repeat_interleave(16,0)).reshape(3,16,3)
        covariance=(offset[:,:,:2].square()+sizes.reshape(3,16,2).square()).mean(1)
        torch.testing.assert_close(covariance,scales.square(),rtol=1e-5,atol=2e-7)
        total=(sizes.prod(-1)*alphas[:,0]).reshape(3,16).sum(1)
        torch.testing.assert_close(total,scales.prod(-1)*opacity[:,0])
        colors=torch.randn(3,3,requires_grad=True)
        (means.sum()+rotations[:,1].sum()+geo.expand_values(colors).sum()).backward()
        torch.testing.assert_close(colors.grad,torch.full((3,3),16.))
        for param in geo.parameters():self.assertTrue(torch.isfinite(param.grad).all())
        clone=GeometryResidual(xyz,args);clone.load_state_dict(geo.state_dict())
        for a,b in zip(geo.prepare(xyz,q,scales,opacity),clone.prepare(xyz,q,scales,opacity)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(geo.diagnostics()['rendered_gaussians'],48)
        self.assertEqual(geo.diagnostics()['source_gaussians'],3)
        old=self.args();old.geometry_subdivide=True
        with self.assertRaises(RuntimeError):GeometryResidual(xyz,old).load_state_dict(geo.state_dict())

    def test_dense_preset_preserves_old_side_and_explicit_cli_priority(self):
        args=self.args();args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        self.assertEqual(args.geometry_subdivision_side,2)
        args.loss_preset='lambertian_geometry_area_dense';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_subdivide);self.assertEqual(args.geometry_subdivision_side,4)
        args.geometry_subdivision_side=2
        apply_loss_preset(args,['--geometry_subdivision_side','2'])
        self.assertEqual(args.geometry_subdivision_side,2)
        args.geometry_subdivision_side=3
        with self.assertRaises(ValueError):GeometryResidual(torch.randn(3,3),args)

    def test_point_light_and_filmic_preset(self):
        args=self.args();args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_filmic';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_point_light);self.assertEqual(args.geometry_tonemap,'filmic')
        light=PointLightTable([0.,1.],torch.tensor([[0.,0.,4.],[2.,0.,4.]]),torch.zeros(3))
        torch.testing.assert_close(light.position(.5),torch.tensor([1.,0.,4.]))
        self.assertFalse(light.raw_directions.requires_grad)
        self.assertEqual(light.render_kwargs(.5)['reference_distance'],8.)
        geo=GeometryResidual(torch.randn(10,3),args)
        # 单调灰阶、有限梯度及checkpoint内自包含显示LUT。
        x=torch.linspace(.001,3,100)[:,None].expand(-1,3).clone().requires_grad_()
        y=apply_filmic_lut(x,geo.display_lut)
        self.assertTrue(((y[1:]-y[:-1])>=-1e-6).all())
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
        clone=GeometryResidual(torch.randn(10,3),args);clone.load_state_dict(geo.state_dict())
        torch.testing.assert_close(clone.display_lut,geo.display_lut)

    def test_shadow_preset_requires_gt_light(self):
        args=self.args();args.render_mode='photometric_lambertian';args.loss_preset='lambertian_geometry_shadow'
        with self.assertRaises(ValueError):apply_loss_preset(args,[])
        args.photometric_light_mode='gt_directional';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_shadows);self.assertTrue(args.geometry_point_light)
        self.assertEqual(args.geometry_tonemap,'filmic')

    def test_palette_groups_chromaticity_and_restores(self):
        args=self.args();args.photometric_material_mode='palette';args.photometric_palette_size=2
        red=torch.tensor([[.8,.1,.1]])*torch.tensor([[.2],[.4],[.8]])
        green=torch.tensor([[.1,.8,.1]])*torch.tensor([[.2],[.4],[.8]])
        colors=linear_to_srgb(torch.cat((red,green)))
        rng=torch.get_rng_state().clone();material=make_material(colors,args)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertEqual(tuple(material.raw_albedo.shape),(2,3))
        self.assertTrue((material.labels[:3]==material.labels[0]).all())
        self.assertTrue((material.labels[3:]==material.labels[3]).all())
        self.assertNotEqual(int(material.labels[0]),int(material.labels[3]))
        material.albedo.sum().backward();self.assertGreater(float(material.raw_albedo.grad.abs().sum()),0)
        restored=make_material(colors,args,material.state_dict())
        torch.testing.assert_close(restored.albedo,material.albedo,rtol=0,atol=0)

    def test_surfel_preset_keeps_legacy_default(self):
        args=self.args();self.assertEqual(args.geometry_shading,'screen')
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_surfel';apply_loss_preset(args,[])
        self.assertEqual(args.geometry_shading,'surfel')
        self.assertTrue(args.geometry_point_light)
        self.assertEqual(GeometryResidual(torch.randn(10,3),args).shading,'surfel')

    def test_lit_preset_uses_reference_visibility_and_palette(self):
        args=self.args();self.assertEqual(args.geometry_photo_visibility_threshold,0.)
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_lit';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_shadows)
        self.assertEqual(args.photometric_material_mode,'palette')
        self.assertEqual(args.geometry_shading,'surfel')
        self.assertEqual(args.geometry_photo_visibility_threshold,.98)
        self.assertEqual(args.geometry_photo_min_shading,.2)

    def test_area_light_power_orientation_and_gradient(self):
        record={'area_shape':'SQUARE','area_size':.0001,'intensity':400.,
                'emission_normal_world':[0.,0.,-1.],'area_axis_u_world':[1.,0.,0.],
                'area_axis_v_world':[0.,1.,0.]}
        light=AreaLightTable([0.],torch.tensor([[0.,0.,10.]]),torch.zeros(3),[record])
        self.assertAlmostEqual(light.reference_intensity,400/(math.pi*64))
        points=torch.zeros(1,3,requires_grad=True);normal=torch.tensor([[0.,0.,1.]])
        kwargs=light.render_kwargs(0.)
        value=area_irradiance(points,normal,kwargs['light_sample_positions'],kwargs['light_emission_normal'],light.reference_intensity,8.)
        torch.testing.assert_close(value,torch.ones_like(value)*4/math.pi,atol=1e-6,rtol=1e-6)
        value.sum().backward();self.assertGreater(float(points.grad.norm()),0)
        back=area_irradiance(points,normal,kwargs['light_sample_positions'],-kwargs['light_emission_normal'],light.reference_intensity,8.)
        self.assertEqual(float(back.sum()),0.)
        legacy=AreaLightTable([0.],torch.tensor([[0.,0.,10.]]),torch.zeros(3),[record],power_convention='legacy')
        self.assertAlmostEqual(legacy.reference_intensity,400/(4*64))
        cutoff=area_irradiance(points,normal,kwargs['light_sample_positions'],kwargs['light_emission_normal'],light.reference_intensity,8.,20.)
        torch.testing.assert_close(cutoff,value*(1-(10/20)**4)**2)

    def test_area_preset_keeps_old_photometric_default(self):
        args=self.args();self.assertFalse(args.geometry_area_emitter)
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area';apply_loss_preset(args,[])
        self.assertTrue(args.geometry_area_emitter)
        self.assertEqual(args.photometric_light_intensity,-1.)

    def test_palette_initialization_source_guard_and_self_contained_restore(self):
        import tempfile,json,hashlib
        import numpy as np
        from pathlib import Path
        args=self.args();args.photometric_material_mode='palette';args.photometric_palette_size=2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args.stage1_model_path=str(root);args.load_iter=1;hashes={}
            for name in ['point_cloud/iteration_1/point_cloud.ply','deform/iteration_1/deform.pth']:
                path=root/name;path.parent.mkdir(parents=True);path.write_bytes(b'source');hashes[str(path)]=hashlib.sha256(b'source').hexdigest()
            path=root/'palette.npz';args.photometric_palette_init=str(path)
            np.savez(path,labels=np.array([0,1,0]),initial_palette=np.array([[.2,.3,.4],[.7,.6,.5]],dtype=np.float32),
                     source_model=str(root),source_iteration=1,source_files_json=json.dumps(hashes))
            colors=torch.rand(3,3);material=make_material(colors,args)
            torch.testing.assert_close(material.albedo[0],torch.tensor([.2,.3,.4]))
            (root/'point_cloud/iteration_1/point_cloud.ply').write_bytes(b'changed')
            with self.assertRaises(ValueError):make_material(colors,args)
            path.unlink()
            restored=make_material(colors,args,material.state_dict())
            torch.testing.assert_close(restored.albedo,material.albedo,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
