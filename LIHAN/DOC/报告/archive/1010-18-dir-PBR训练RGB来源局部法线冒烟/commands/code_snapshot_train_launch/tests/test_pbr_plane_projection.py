"""原生几何投影与全部135时间来源核验；不修改评分或使用GT。"""
import tempfile,unittest,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from argparse import ArgumentParser
import numpy as np
import torch
from plyfile import PlyData,PlyElement
from scene.pbr_geometry import quaternion_normal
from scene.pbr_plane_projection import (PlaneProjection,align_quaternion_normal,create_plane_projection,
    load_plane_projection,validate_plane_projection,ASSET,METADATA)
from arguments.stage2 import Stage2PipelineParams,Stage2OptimizationParams
from scripts.loss_stage2 import apply_loss_preset


class DummyDeform:
    def __init__(self):self.deform=torch.nn.Linear(1,1,bias=False)
    def step(self,xyz,time,**kwargs):
        dx=torch.zeros_like(xyz);dx[0,0]=time[0,0]*.1
        return dict(d_xyz=dx,d_rotation=xyz.new_zeros((len(xyz),4)))


class PBRPlaneProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.model=Path(self.temp.name)
        x,y=np.meshgrid(np.linspace(-1,1,8),np.linspace(-1,1,8));x=x.flatten();y=y.flatten()
        xyz=np.column_stack((x,y,.1+.1*x+.2*y)).astype('f4');q=np.zeros((64,4),dtype='f4');q[:,0]=1
        fields=[('x','f4'),('y','f4'),('z','f4')]+[(f'rot_{i}','f4') for i in range(4)]+[('opacity','f4'),('fea_0','f4')]
        v=np.empty(64,dtype=fields)
        for i,k in enumerate(['x','y','z']):v[k]=xyz[:,i]
        for i in range(4):v[f'rot_{i}']=q[:,i]
        v['opacity']=2;v['fea_0']=-2
        ply=self.model/'point_cloud/iteration_35000/point_cloud.ply';ply.parent.mkdir(parents=True);PlyData([PlyElement.describe(v,'vertex')]).write(ply)
        self.deform=DummyDeform();pth=self.model/'deform/iteration_35000/deform.pth';pth.parent.mkdir(parents=True);torch.save(self.deform.deform.state_dict(),pth)
        source={'/source/'+str(p.relative_to(self.model)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ply,pth]}
        (self.model/'stage1_source.json').write_text(json.dumps(source))
        self.pc=SimpleNamespace(get_xyz=torch.nn.Parameter(torch.tensor(xyz)),_rotation=torch.nn.Parameter(torch.tensor(q)))
        self.cameras=[SimpleNamespace(image_name_train_light=f'{i:04}',fid=torch.tensor([t]),camera_center=torch.tensor([0.,0.,3.])) for i,t in enumerate([.1,.5,.9])]
        self.args=SimpleNamespace(pbr_plane_projection=True,pbr_deformation_anchor=True,pbr_optimize_geometry=True,
             render_mode='original_ir',pbr_light_mode='gt_directional',geometry_source_deformation_mode='dynamic',
             pbr_plane_band=.08,pbr_plane_support_band=.2,pbr_plane_motion_span=.01)

    def create(self):return create_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,35000)

    def test_roundtrip_filters_temporally_moving_source_point(self):
        p,r=self.create();self.assertFalse(bool(p.support[0]));self.assertEqual(int(p.support.sum()),63)
        q=load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,35000)
        self.assertTrue(torch.equal(p.support,q.support));self.assertFalse(r['gt_geometry_inputs'])

    def test_projection_changes_real_geometry_and_preserves_other_predictions(self):
        support=torch.tensor([True,False]);plane=torch.tensor([0.,0.,1.]);offset=torch.tensor(0.)
        proj=PlaneProjection(support,plane,offset,torch.ones(2))
        xyz=torch.nn.Parameter(torch.tensor([[.1,.2,.3],[.4,.5,.6]]));q=torch.nn.Parameter(torch.tensor([[1.,.2,.3,.1],[1.,.4,.2,.1]]))
        pc=SimpleNamespace(get_xyz=xyz,_rotation=q);dx=torch.ones((2,3))*.05;dr=torch.ones((2,4))*.01
        pred=dict(d_xyz=dx,d_rotation=dr,d_color=torch.ones((2,6)),d_scaling=torch.zeros((2,2)),d_opacity=None)
        out=proj.apply(pc,pred);effective=xyz+out['d_xyz'];n=quaternion_normal(q+out['d_rotation'])
        torch.testing.assert_close(effective[0,2],torch.tensor(0.),atol=1e-7,rtol=0)
        torch.testing.assert_close(n[0],plane,atol=1e-6,rtol=0)
        torch.testing.assert_close(out['d_xyz'][1],dx[1],atol=0,rtol=0)
        torch.testing.assert_close(out['d_rotation'][1],dr[1],atol=0,rtol=0)
        self.assertIs(out['d_color'],pred['d_color']);self.assertIs(out['d_opacity'],None)
        self.assertTrue(torch.equal(pred['d_xyz'],dx))
        (effective.square().sum()+(q+out['d_rotation']).square().sum()).backward()
        self.assertTrue(torch.isfinite(xyz.grad).all() and torch.isfinite(q.grad).all())
        self.assertAlmostEqual(float(xyz.grad[0,2]),0.,places=6)
        self.assertGreater(float(xyz.grad[0,:2].abs().sum()),0.)

    def test_quaternion_antipodal_and_raw_norm(self):
        q=torch.tensor([[2.,0.,0.,0.],[1.,0.,0.,0.]])
        target=torch.tensor([[0.,0.,-1.],[1.,0.,0.]])
        out=align_quaternion_normal(q,target)
        torch.testing.assert_close(quaternion_normal(out),target,atol=1e-6,rtol=0)
        torch.testing.assert_close(out.norm(dim=1),q.norm(dim=1),atol=1e-6,rtol=0)
        with self.assertRaises(ValueError):align_quaternion_normal(torch.zeros((2,4)),target)

    def test_source_xyz_and_asset_overwrite_rejected(self):
        with torch.no_grad():self.pc.get_xyz.add_(.01)
        with self.assertRaises(ValueError):self.create()
        with torch.no_grad():self.pc.get_xyz.sub_(.01)
        # 浮点加减不保证逐位恢复，直接重新读取真实来源。
        v=PlyData.read(self.model/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
        with torch.no_grad():self.pc.get_xyz.copy_(torch.tensor(np.column_stack([v[k] for k in ['x','y','z']])))
        self.create()
        with self.assertRaises(FileExistsError):self.create()

    def test_asset_fake_support_rejected_after_sha_rewrite(self):
        self.create();path=self.model/ASSET
        with np.load(path) as src:arrays={k:src[k].copy() for k in src.files}
        arrays['support'][0]=True;np.savez(path,**arrays)
        meta=json.loads((self.model/METADATA).read_text());meta['asset_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(self.model/METADATA).write_text(json.dumps(meta))
        with self.assertRaises(ValueError):load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,35000)

    def test_marker_binding_required_for_trained_iteration(self):
        _,r=self.create()
        with self.assertRaises(FileNotFoundError):load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,55000)
        path=self.model/'pbr_directional/iteration_55000/lighting.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(iteration=55000,plane_projection_sha256='wrong')))
        with self.assertRaises(ValueError):load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,55000)
        path.write_text(json.dumps(dict(iteration=55000,plane_projection_sha256=r['asset_sha256'])))
        load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,55000)

    def test_source_network_change_rejected(self):
        self.create()
        with torch.no_grad():self.deform.deform.weight.add_(.01)
        with self.assertRaises(ValueError):load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,35000)

    def test_camera_change_rejected(self):
        self.create();self.cameras[1].camera_center[0]=.01
        with self.assertRaises(ValueError):load_plane_projection(self.model,self.pc,self.deform,self.cameras,self.args,35000)

    def test_invalid_mode_and_thresholds(self):
        for key,value in [('pbr_deformation_anchor',False),('pbr_plane_motion_span',float('nan')),
                          ('pbr_plane_support_band',.01),('render_mode','photometric_lambertian')]:
            original=getattr(self.args,key);setattr(self.args,key,value)
            with self.assertRaises(ValueError):validate_plane_projection(self.args)
            setattr(self.args,key,original)

    def test_preset_and_default_compatibility(self):
        parser=ArgumentParser();Stage2PipelineParams(parser);Stage2OptimizationParams(parser)
        for argv in [['--loss_preset','pbr_gt_directional_plane'],['--pbr_light_mode','gt_directional','--pbr_plane_projection']]:
            args=parser.parse_args(argv);apply_loss_preset(args,argv);self.assertEqual(args.loss_preset,'pbr_gt_directional_plane');validate_plane_projection(args)
            self.assertTrue(args.pbr_deformation_anchor and args.pbr_albedo_interior_init and args.pbr_optimize_geometry)
        for preset in ['auto','irgs_baseline','pbr_gt_directional','pbr_gt_directional_geometry','pbr_gt_directional_refine','pbr_gt_directional_anchor']:
            argv=['--loss_preset',preset];args=parser.parse_args(argv);apply_loss_preset(args,argv);self.assertFalse(args.pbr_plane_projection)
