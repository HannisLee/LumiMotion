"""固定TRAIN稀疏RGB位置到原生Median邻像素三维表面的距离。"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

ASSET_SHA256='d8c8f2909ccfedfcb30b28c53d700aa2b82eab517af16ce77828e64839142e3d'
TRACKS_SHA256='1f73f399bc0fce74762a0ced282d8c4283b0efaed9172122f1389a27901936ca'


def validate_asset(path):
    path=Path(path).resolve();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ASSET_SHA256:
        raise ValueError('固定稀疏Median资产SHA不符；不得重选目标或像素。')
    asset=json.loads(raw)
    assert asset['version']==1 and asset['gt_read'] is False and asset['init_weights_loaded'] is False
    assert asset['held_used_as_targets'] is False and asset['parent_tracks_sha256']==TRACKS_SHA256
    assert asset['track_count']==len(asset['targets'])==115 and asset['fit_observation_count']==600
    assert asset['camera_count']==len(asset['frames'])==135 and asset['camera_extent']==4.290023326873779
    assert asset['coefficient']==.1
    counts=np.zeros(115,dtype=int)
    for name,frame in asset['frames'].items():
        assert frame['fit_view']==(frame['source_index']%9!=4)
        if not frame['fit_view']:assert not frame['observations']
        for obs in frame['observations']:
            j=obs['target_index'];counts[j]+=1
            assert obs['track_id']==asset['targets'][j]['id']
            p=obs['pixels_yx'];assert 4<=len(p)<=8 and len({tuple(q) for q in p})==len(p)
            assert all(0<=y<400 and 0<=x<400 for y,x in p)
            assert np.asarray(obs['rays_world']).shape==(len(p),3)
    assert int(counts.sum())==600 and np.array_equal(counts,[t['fit_observations'] for t in asset['targets']])
    return asset


def pixel_surface_loss(depth,packed,extent):
    """全部观测固定分母；无效/截断项为extent且没有伪造梯度。"""
    assert depth.shape==(400,400) and extent>0
    d=depth.reshape(-1)[packed['indices']]
    valid=packed['member'] & torch.isfinite(d) & (d>0)
    clean=torch.where(valid,d,torch.zeros_like(d))
    points=packed['origin'][None,None,:]+clean[:,:,None]*packed['rays']
    # 数值epsilon为1微米距离，仅避免零处sqrt导数；不是可调损失权重。
    distance=torch.sqrt((points-packed['targets'][:,None,:]).square().sum(-1)+1e-12)
    distance=torch.where(valid,distance,depth.new_full((),extent))
    minimum=distance.min(dim=1).values.clamp_max(extent)
    loss=(135./115.)*(minimum/extent*packed['inverse_views']).sum()
    diagnostics=dict(observations=len(minimum),missing_observations=(~valid.any(1)).sum().detach(),
                     capped_observations=(minimum>=extent).sum().detach(),
                     mean_nearest_distance_m=minimum.mean().detach(),
                     unweighted_track_equal_estimate=loss.detach())
    return loss,diagnostics


class FixedSparseRGBSurface:
    def __init__(self,path,extent):
        self.path=Path(path).resolve();self.asset=validate_asset(self.path)
        self.extent=float(extent)
        assert self.extent==self.asset['camera_extent']
        self.cache={};self.checked_cameras=set()

    def metadata(self):
        return dict(asset_path=str(self.path),asset_sha256=ASSET_SHA256,tracks=115,fit_observations=600,
                    held_targets=0,native_depth='median',maximum_fixed_pixels=8,
                    camera_weight='135/115 and inverse original fit observations per track',
                    invalid_and_clipped_distance_m=self.extent,coefficient=.1,
                    screen_route='original_photo_proxy_then_original_expected_normal_vector_merge',
                    native_opacity_route='original_Median_backward_without_extra_detach',GT_read=False)

    def _check_camera(self,camera):
        name=camera.image_name_train_light
        if name in self.checked_cameras:return
        frame=self.asset['frames'][name]
        assert (camera.image_width,camera.image_height)==(400,400)
        E=camera.world_view_transform.detach().cpu().numpy().T
        assert np.array_equal(E,np.array(frame['E'],dtype=np.float32))
        fx=400/(2*math.tan(camera.FoVx/2));fy=400/(2*math.tan(camera.FoVy/2))
        assert max(abs(fx-frame['K400'][0][0]),abs(fy-frame['K400'][1][1]))<1e-7
        self.checked_cameras.add(name)

    def _pack(self,name,device,dtype):
        key=(name,str(device),dtype)
        if key in self.cache:return self.cache[key]
        frame=self.asset['frames'][name];observations=frame['observations']
        if not observations:self.cache[key]=None;return None
        n=len(observations);indices=np.zeros((n,8),np.int64);member=np.zeros((n,8),bool)
        rays=np.zeros((n,8,3));targets=np.zeros((n,3));inverse=np.zeros(n)
        for row,obs in enumerate(observations):
            j=obs['target_index'];target=self.asset['targets'][j];p=obs['pixels_yx'];m=len(p)
            indices[row,:m]=[y*400+x for y,x in p];member[row,:m]=True
            rays[row,:m]=obs['rays_world'];targets[row]=target['xyz'];inverse[row]=1/target['fit_observations']
        packed={k:torch.as_tensor(v,device=device,dtype=(torch.long if k=='indices' else torch.bool if k=='member' else dtype))
                for k,v in dict(indices=indices,member=member,rays=rays,targets=targets,inverse_views=inverse,
                                origin=np.array(frame['origin_world'])).items()}
        self.cache[key]=packed;return packed

    def __call__(self,camera,median_depth):
        self._check_camera(camera)
        assert median_depth.shape==(1,400,400)
        packed=self._pack(camera.image_name_train_light,median_depth.device,median_depth.dtype)
        if packed is None:
            return median_depth.new_zeros(()),dict(observations=0,missing_observations=0,capped_observations=0,
                      mean_nearest_distance_m=0.,unweighted_track_equal_estimate=0.)
        return pixel_surface_loss(median_depth[0],packed,self.extent)
