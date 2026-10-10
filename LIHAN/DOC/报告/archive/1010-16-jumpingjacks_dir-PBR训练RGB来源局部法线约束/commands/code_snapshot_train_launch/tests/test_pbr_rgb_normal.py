"""RGB来源约束、局部运输、legacy隔离和checkpoint协议。"""
import hashlib,json,tempfile,unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData,PlyElement
from arguments.stage2 import Stage2PipelineParams,Stage2OptimizationParams
from scripts.loss_stage2 import apply_loss_preset,resolve_preset
from scene.pbr_rgb_normal import (SourceLocalRGBNormal,transport_quaternion,validate_rgb_normal,
    create_rgb_normal,load_rgb_normal,INPUT_METHOD,METADATA,ASSET,INPUT_ASSET,INPUT_METADATA)
from scene.pbr_geometry import quaternion_normal
def build_rotation(q):
    w,x,y,z=torch.nn.functional.normalize(q,dim=-1).unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],dim=-1).reshape(-1,3,3)



class RGBNormalTest(unittest.TestCase):
    def args(self):
        p=ArgumentParser();Stage2PipelineParams(p);Stage2OptimizationParams(p)
        a=p.parse_args([]);a.load_iter=35000;a.iterations=55000
        a.loss_preset='pbr_gt_directional_source_rgb_normal';a.pbr_rgb_normal_prior_path='input'
        apply_loss_preset(a,[])
        return a

    def operator(self):
        torch.manual_seed(1010)
        q=torch.randn(8,4);q[:,0]+=2
        target=torch.nn.functional.normalize(torch.randn(8,3),dim=-1)
        confidence=torch.tensor([1.,0.,1.,0.,1.,0.,1.,1.])
        return SourceLocalRGBNormal(q,target,confidence),q

    def fixture(self,path):
        model=Path(path)/'model';model.mkdir()
        ply=model/'point_cloud/iteration_35000/point_cloud.ply';ply.parent.mkdir(parents=True)
        pth=model/'deform/iteration_35000/deform.pth';pth.parent.mkdir(parents=True)
        v=np.zeros(5,dtype=[(k,'<f4') for k in ['x','y','z','rot_0','rot_1','rot_2','rot_3']]);v['rot_0']=1
        PlyData([PlyElement.describe(v,'vertex')]).write(str(ply))
        deform=SimpleNamespace(deform=torch.nn.Linear(3,4));deform.deform.requires_grad_(False);torch.save(deform.deform.state_dict(),pth)
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (model/'stage1_source.json').write_text(json.dumps({str(p):sha(p) for p in [ply,pth]}))
        q=torch.tensor(np.column_stack([v[f'rot_{i}'] for i in range(4)]))
        pc=SimpleNamespace(_rotation=torch.nn.Parameter(q),get_xyz=torch.zeros(5,3))
        source=Path(path)/'data';(source/'images').mkdir(parents=True)
        keys=[f'{i:04d}' for i in list(range(1,10))+[11,12,13]]
        frames=[];rgb={}
        for k in keys:
            p=source/'images'/(k+'.png');Image.new('RGBA',(4,4),(120,120,120,255)).save(p)
            frames.append(dict(source_frame=int(k),file_path=k+'.png'))
            rgb[k]=dict(path=str(p),sha256=sha(p))
        train=source/'transforms_train.json';train.write_text(json.dumps(dict(frames=frames)))
        directions=source/'directions.json';directions.write_text(json.dumps({'frames':{k:dict(light_dir_world=[0.,0.,1.],intensity=1.,light_rgb=[1.,1.,1.]) for k in keys}}))
        plane=model/'pbr_plane_projection.npz';support=np.array([True,False,False,False,False]);np.savez(plane,support=support)
        args=self.args();args.source_path=str(source);args.train_light_folder='images';args.pbr_gt_lights_path=str(directions)
        prior=Path(path)/'input';prior.mkdir();args.pbr_rgb_normal_prior_path=str(prior)
        gram=np.tile(np.diag([5.,4.,3.]).astype(np.float32),(5,1,1));vector=np.tile(np.array([.1,.03,.9],np.float32),(5,1))
        rhs=np.einsum('nij,nj->ni',gram,vector);y2=np.sum(rhs*vector,axis=-1)
        fitted=np.linalg.solve(gram+np.eye(3,dtype=np.float32)[None]*1e-5,rhs[...,None])[...,0]
        target=fitted/np.linalg.norm(fitted,axis=-1,keepdims=True);target[0]=[0,0,1]
        confidence=(~support).astype(np.float32)
        asset=prior/'training_rgb_normal_constraint.npz'
        np.savez(asset,canonical_axis=target.astype(np.float32),confidence=confidence,observations=np.full(5,12,np.int32),
            gram_eigenvalues=np.linalg.eigvalsh(gram),relative_luminance_residual=np.zeros(5,np.float32),
            source_canonical_axis=np.tile(np.array([0.,0.,1.],np.float32),(5,1)),gram=gram,rhs=rhs,y_squared=y2)
        source_manifest={k:dict(path=str(p),sha256=sha(p)) for k,p in dict(PLY=ply,MLP=pth,TRAIN_cameras=train,directions_only=directions,plane_arrays=plane).items()}
        meta=dict(format_version=1,method=INPUT_METHOD,points=5,TRAIN_frames=12,training_frame_keys=keys,
            TRAIN_RGBA_manifest=rgb,TEST_frames_used=0,optimizer_steps=0,native_feature_probability_exact=True,
            GT_labels_read=False,GT_light_photometry_read=False,learned_40000_geometry_material_feature_used=False,
            production_pipeline_changed=False,full_stage2_started=False,asset_sha256=sha(asset),source_manifest=source_manifest,
            confidence_points=4,threshold_protocol=dict(min_observations=12,gram_eigenvalue_ratio=.01,min_coefficient_norm=.01,
                max_relative_residual=.3,source_expected_Z_distance_m=.025,own_alpha_min=.99,source_raster_alpha_min=.9))
        (prior/'training_rgb_normal_constraint.json').write_text(json.dumps(meta))
        cameras=[SimpleNamespace(image_name_train_light=k) for k in keys]
        return model,pc,deform,cameras,args

    def test_preset_auto_and_legacy(self):
        a=self.args();self.assertEqual(a.lambda_pbr_rgb_normal_prior,2.)
        self.assertTrue(a.pbr_source_local_rotation);self.assertEqual(a.pbr_feature_gate_parameterization,'probability')
        a.loss_preset='auto';self.assertEqual(resolve_preset(a,'original_ir'),'pbr_gt_directional_source_rgb_normal')
        a.pbr_source_local_rotation=False;a.pbr_rgb_normal_prior_path='';a.lambda_pbr_rgb_normal_prior=0
        self.assertEqual(resolve_preset(a,'original_ir'),'pbr_gt_directional_plane_native_direct_probability_gate')
        validate_rgb_normal(a)

    def test_invalid_configuration(self):
        for field,value in [('lambda_pbr_rgb_normal_prior',0),('lambda_pbr_rgb_normal_prior',float('nan')),
            ('pbr_rgb_normal_prior_path',''),('render_mode','photometric_lambertian'),('pbr_rasterizer_backend','installed')]:
            a=self.args();setattr(a,field,value)
            with self.assertRaises(ValueError):validate_rgb_normal(a)

    def test_initial_raw_deformation_exact(self):
        op,q=self.operator();pc=SimpleNamespace(_rotation=q.clone().requires_grad_());raw=torch.randn_like(q)
        out=op.apply(pc,dict(d_rotation=raw,d_xyz=torch.randn(8,3)))
        self.assertTrue(torch.equal(out['d_rotation'],raw))

    def test_rotation_matches_source_matrix_transport(self):
        op,q=self.operator();current=q+torch.randn_like(q)*.2;raw=torch.randn_like(q)*.3
        out=op.apply(SimpleNamespace(_rotation=current),dict(d_rotation=raw))
        actual=quaternion_normal(current+out['d_rotation'])
        expected=torch.einsum('nij,nj->ni',build_rotation(q+raw),
            torch.einsum('nij,nj->ni',build_rotation(q).transpose(1,2),quaternion_normal(current)))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=0)

    def test_world_transport_preserves_raw_norm(self):
        op,q=self.operator();current=q+torch.randn_like(q)*.2;raw=torch.randn_like(q)*.1
        moved=transport_quaternion(q+raw,q,current)
        torch.testing.assert_close(moved.norm(dim=-1),(q+raw).norm(dim=-1),atol=1e-6,rtol=1e-6)

    def test_prior_gradient_only_eligible_full_points_preserved(self):
        op,q=self.operator();pc=SimpleNamespace(_rotation=q.clone().requires_grad_());op.loss(pc).backward()
        self.assertTrue(torch.isfinite(pc._rotation.grad).all());self.assertGreater(float(pc._rotation.grad.norm()),0)
        self.assertTrue((pc._rotation.grad[op.confidence==0]==0).all());self.assertEqual(len(pc._rotation),8)

    def test_local_forward_has_geometry_and_raw_gate_gradients(self):
        op,q=self.operator();current=q.clone().requires_grad_();raw=(torch.randn_like(q)*.3).requires_grad_()
        d=op.apply(SimpleNamespace(_rotation=current),dict(d_rotation=raw))
        loss=(quaternion_normal(current+d['d_rotation'])*op.target).sum();loss.backward()
        for grad in [current.grad,raw.grad]:self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_other_deformation_buffers_unchanged(self):
        op,q=self.operator();pred=dict(d_rotation=torch.randn_like(q),d_xyz=torch.randn(8,3),d_color=torch.randn(8,3))
        out=op.apply(SimpleNamespace(_rotation=q),pred)
        self.assertIs(out['d_xyz'],pred['d_xyz']);self.assertIs(out['d_color'],pred['d_color'])

    def test_invalid_arrays_rejected(self):
        op,q=self.operator()
        for source,target,confidence in [(q*0,op.target,op.confidence),(q,op.target*0,op.confidence),
            (q,op.target,op.confidence*.1),(q[:1],op.target,op.confidence)]:
            with self.assertRaises(ValueError):SourceLocalRGBNormal(source,target,confidence)

    def test_roundtrip_and_checkpoint_marker(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path);op=create_rgb_normal(model,pc,deform,cameras,args,35000)
            loaded=load_rgb_normal(model,pc,deform,cameras,args,35000)
            self.assertTrue(torch.equal(op.source,loaded.source));self.assertTrue(torch.equal(op.target,loaded.target))
            marker=model/'pbr_directional/iteration_55000/lighting.json';marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps(dict(rgb_normal_metadata_sha256=op.metadata_sha)))
            load_rgb_normal(model,pc,deform,cameras,args,55000)
            marker.write_text('{}')
            with self.assertRaises(ValueError):load_rgb_normal(model,pc,deform,cameras,args,55000)

    def test_saved_asset_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path);create_rgb_normal(model,pc,deform,cameras,args,35000)
            (model/ASSET).write_bytes(b'tampered')
            with self.assertRaises(ValueError):load_rgb_normal(model,pc,deform,cameras,args,35000)

    def test_source_nn_change_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path)
            with torch.no_grad():deform.deform.weight.add_(.1)
            with self.assertRaises(ValueError):create_rgb_normal(model,pc,deform,cameras,args,35000)

    def test_training_image_change_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path);create_rgb_normal(model,pc,deform,cameras,args,35000)
            Image.new('RGBA',(4,4),(1,1,1,255)).save(Path(args.source_path)/'images/0001.png')
            with self.assertRaises(ValueError):load_rgb_normal(model,pc,deform,cameras,args,35000)

    def test_GT_label_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path);meta=Path(args.pbr_rgb_normal_prior_path)/'training_rgb_normal_constraint.json'
            data=json.loads(meta.read_text());data['GT_labels_read']=True;meta.write_text(json.dumps(data))
            with self.assertRaises(ValueError):create_rgb_normal(model,pc,deform,cameras,args,35000)

    def test_candidate_fit_tamper_rejected_even_new_asset_sha(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform,cameras,args=self.fixture(path);root=Path(args.pbr_rgb_normal_prior_path);asset=root/'training_rgb_normal_constraint.npz'
            with np.load(asset) as values:data={k:values[k].copy() for k in values.files}
            data['canonical_axis'][1]=[1,0,0];np.savez(asset,**data)
            meta=root/'training_rgb_normal_constraint.json';record=json.loads(meta.read_text());record['asset_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest();meta.write_text(json.dumps(record))
            with self.assertRaises(ValueError):create_rgb_normal(model,pc,deform,cameras,args,35000)


if __name__=='__main__':unittest.main()
