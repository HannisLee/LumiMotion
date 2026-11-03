"""T01 只读 SH 提取：仅读取帧编号、相机平移、冻结 PLY 和 observations。"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from plyfile import PlyData
from LIHAN.scripts.static_rgb_light_shbg_t01_core import sh_luminance, validate_alignment


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def camera_fields(path):
    """扫描 JSON，只解析 frames 数组中 file_path 和 transform_matrix。

    对所有其他字段跳过文本，不解析其值。返回相机世界坐标和平移对应帧。
    """
    s=Path(path).read_text(); dec=json.JSONDecoder(); length=len(s)
    def ws(i):
        while i<length and s[i].isspace(): i+=1
        return i
    def string(i): return dec.raw_decode(s,i)
    def skip(i):
        i=ws(i)
        if s[i]=='"': return string(i)[1]
        if s[i] in '[{':
            stack=[s[i]]; i+=1
            while stack:
                if s[i]=='"': i=string(i)[1]; continue
                if s[i] in '[{': stack.append(s[i])
                elif s[i] in ']}':
                    if (stack.pop(),s[i]) not in [('[',']'),('{','}')]: raise ValueError('非法 JSON')
                i+=1
            return i
        while i<length and s[i] not in ',}] \t\r\n': i+=1
        return i
    def frame(i):
        i=ws(i)
        if s[i]!='{': raise ValueError('帧对象格式错误')
        i+=1; got={}
        while True:
            i=ws(i)
            if s[i]=='}': return got,i+1
            key,i=string(i); i=ws(i)
            if s[i]!=':': raise ValueError('JSON 缺少冒号')
            i=ws(i+1)
            if key in ('file_path','transform_matrix'):
                if key in got: raise ValueError('相机字段重复')
                got[key],i=dec.raw_decode(s,i)
            else: i=skip(i)
            i=ws(i)
            if s[i]==',': i+=1
            elif s[i]!='}': raise ValueError('帧对象格式错误')
    i=ws(0)
    if s[i]!='{': raise ValueError('相机 JSON 根节点错误')
    i+=1; frames=None
    while True:
        i=ws(i)
        if s[i]=='}': break
        key,i=string(i); i=ws(i)
        if s[i]!=':': raise ValueError('JSON 缺少冒号')
        i=ws(i+1)
        if key=='frames':
            if frames is not None or s[i]!='[': raise ValueError('frames 重复或格式错误')
            frames=[]; i+=1
            while True:
                i=ws(i)
                if s[i]==']': i+=1; break
                item,i=frame(i); frames.append(item); i=ws(i)
                if s[i]==',': i+=1
                elif s[i]!=']': raise ValueError('frames 格式错误')
        else: i=skip(i)
        i=ws(i)
        if s[i]==',': i+=1
        elif s[i]!='}': raise ValueError('根节点格式错误')
    if frames is None: raise ValueError('缺少 frames')
    rows=[]
    for item in frames:
        if set(item)!={'file_path','transform_matrix'}: raise ValueError('缺少相机字段')
        match=re.search(r'_(\d+)$',Path(item['file_path']).stem)
        if match is None: raise ValueError('帧号格式错误')
        matrix=np.asarray(item['transform_matrix'],dtype=np.float64)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all(): raise ValueError('相机矩阵无效')
        rows.append((int(match.group(1)),matrix[:3,3]))
    rows.sort(key=lambda x:x[0]); ids=np.array([r[0] for r in rows]); centers=np.array([r[1] for r in rows])
    if not np.array_equal(ids,np.arange(1,151)): raise ValueError('相机帧不匹配')
    return ids,centers


def read_coefficients(ply_path, ids):
    """与 GaussianModel.load_ply 的 albedo DC/rest 通道布局一致。"""
    vertex=PlyData.read(str(ply_path))['vertex']; names=vertex.data.dtype.names
    rest=sorted((n for n in names if n.startswith('albedo_rest_')),key=lambda n:int(n.split('_')[-1]))
    if len(rest)!=45: raise ValueError('Stage 1 SH 阶数与模型不符')
    if np.any(ids<0) or np.any(ids>=len(vertex)): raise ValueError('Gaussian ID 越界')
    dc=np.stack([vertex[f'albedo_dc_{i}'][ids] for i in range(3)],axis=1)[:,:,None]
    tail=np.stack([vertex[name][ids] for name in rest],axis=1).reshape(len(ids),3,15)
    return np.concatenate([dc,tail],axis=2).astype(np.float64)


def extract(observations,source,model,iteration,dest,expected_obs_sha,expected_ids):
    if digest(observations)!=expected_obs_sha: raise ValueError('observations SHA256 不匹配')
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=False)
    with np.load(observations,allow_pickle=False) as d:
        ids=d['gaussian_ids'].copy(); frames=d['frames'].copy(); xyz=d['frame_xyz'].copy(); y=d['intensity'].copy(); w=d['weights'].copy()
    validate_alignment(ids,expected_ids,frames)
    if xyz.shape!=(150,len(ids),3) or y.shape!=(150,len(ids)) or w.shape!=(150,len(ids)): raise ValueError('观测尺寸错误')
    cf=Path(source)/'transforms_all.json'; cframes,centers=camera_fields(cf)
    if not np.array_equal(frames,cframes): raise ValueError('相机和观测帧不一致')
    ply=Path(model)/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    coeff=read_coefficients(ply,ids); valid=np.isfinite(y)&(w>0); result={'frames':frames,'gaussian_ids':ids}; diagnostics={}
    for L in (0,1,2):
        h,raw=sh_luminance(coeff,xyz,centers,L); clipped=(raw<0)|(raw>1)
        fractions=clipped[valid].mean(axis=0); diagnostics[str(L)]={'low':(raw[valid]<0).mean(axis=0).tolist(),'high':(raw[valid]>1).mean(axis=0).tolist(),'invalid_clipping':bool((fractions>.2).any())}
        result[f'h_L{L}']=h; result[f'raw_L{L}']=raw
    np.savez_compressed(dest/'sh_values.npz',**result)
    (dest/'extraction.json').write_text(json.dumps({'input_sha256':digest(observations),'camera_sha256':digest(cf),'ply_sha256':digest(ply),'gt_read':False,'diagnostics':diagnostics},indent=2))
    return diagnostics
