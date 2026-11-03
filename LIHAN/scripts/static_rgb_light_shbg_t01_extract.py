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


def camera_fields(path, include_matrices=False):
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
            if key in ('file_path','transform_matrix','source_frame'):
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
        if set(item)!={'file_path','transform_matrix','source_frame'}: raise ValueError('缺少相机字段')
        match=re.search(r'_(\d+)$',Path(item['file_path']).stem)
        if match is None or int(match.group(1))!=item['source_frame']: raise ValueError('帧号格式错误')
        matrix=np.asarray(item['transform_matrix'],dtype=np.float64)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all(): raise ValueError('相机矩阵无效')
        rows.append((int(match.group(1)),matrix))
    rows.sort(key=lambda x:x[0]); ids=np.array([r[0] for r in rows]); matrices=np.array([r[1] for r in rows]); centers=matrices[:,:3,3]
    if not np.array_equal(ids,np.arange(1,151)): raise ValueError('相机帧不匹配')
    return (ids,centers,matrices) if include_matrices else (ids,centers)


def raw_camera_fields(path):
    """仅解析 frames.*.frame 与 frames.*.extrinsics.camera_to_world。"""
    s=Path(path).read_text(); dec=json.JSONDecoder(); N=len(s)
    def ws(i):
        while i<N and s[i].isspace(): i+=1
        return i
    def skip(i):
        i=ws(i)
        if s[i]=='"': return dec.raw_decode(s,i)[1]
        if s[i] in '[{':
            stack=[s[i]]; i+=1
            while stack:
                if s[i]=='"': i=dec.raw_decode(s,i)[1]; continue
                if s[i] in '[{': stack.append(s[i])
                elif s[i] in ']}':
                    if (stack.pop(),s[i]) not in [('[',']'),('{','}')]: raise ValueError('非法 JSON')
                i+=1
            return i
        while i<N and s[i] not in ',}] \t\r\n': i+=1
        return i
    def obj(i,level):
        i=ws(i)
        if s[i]!='{': raise ValueError('原始相机对象格式错误')
        i+=1; result={}
        while True:
            i=ws(i)
            if s[i]=='}': return result,i+1
            key,i=dec.raw_decode(s,i); i=ws(i)
            if s[i]!=':': raise ValueError('原始相机格式错误')
            i=ws(i+1)
            nested=(level=='root' and key=='frames') or (level=='frames' and re.fullmatch(r'\d{4}',key)) or (level=='frame' and key=='extrinsics')
            if nested:
                if key in result: raise ValueError('重复相机字段')
                nxt={'root':'frames','frames':'frame','frame':'extrinsics'}[level]
                result[key],i=obj(i,nxt)
            elif (level=='frame' and key=='frame') or (level=='extrinsics' and key=='camera_to_world'):
                if key in result: raise ValueError('重复相机字段')
                result[key],i=dec.raw_decode(s,i)
            else: i=skip(i)
            i=ws(i)
            if s[i]==',': i+=1
            elif s[i]!='}': raise ValueError('原始相机格式错误')
    raw,_=obj(0,'root'); frames=raw.get('frames',{}); ids=[]; mats=[]
    for key in sorted(frames):
        entry=frames[key]; number=int(key)
        if entry.get('frame')!=number or set(entry.get('extrinsics',{}))!={'camera_to_world'}: raise ValueError('原始相机帧字段不完整')
        matrix=np.asarray(entry['extrinsics']['camera_to_world'],dtype=np.float64)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all(): raise ValueError('原始相机矩阵无效')
        ids.append(number); mats.append(matrix)
    ids=np.asarray(ids); mats=np.asarray(mats)
    if not np.array_equal(ids,np.arange(1,151)): raise ValueError('原始相机帧不完整')
    return ids,mats


def check_provenance(source,derived):
    """检查上游准备命令、原始相机及转换文件的冻结摘要。"""
    command_path=derived.parent/'preparation_commands.json'; commands=json.loads(command_path.read_text())
    cmd=commands[0]; project=derived.parents[2]
    if '--source' not in cmd or '--output' not in cmd: raise ValueError('准备命令缺少输入输出')
    if (project/cmd[cmd.index('--source')+1]).resolve()!=source.resolve() or (project/cmd[cmd.index('--output')+1]).resolve()!=derived.resolve(): raise ValueError('准备命令来源不符')
    text=(derived/'rgb_only_manifest.json').read_text()
    def value(key):
        pattern=r'"'+re.escape(key)+r'"\s*:\s*"([0-9a-f]{64})"'
        found=re.findall(pattern,text)
        if len(found)!=1: raise ValueError('来源清单缺少或重复 '+key)
        return found[0]
    if value('camera_sha256')!=digest(source/'camera.json') or value('transforms_all.json')!=digest(derived/'transforms_all.json'): raise ValueError('相机来源摘要不匹配')
    return cmd


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
    derived=Path(observations).resolve().parents[1]/'source_all'
    cf=derived/'transforms_all.json'; cframes,centers,matrices=camera_fields(cf,include_matrices=True)
    original_frames,original=raw_camera_fields(Path(source)/'camera.json')
    if not np.array_equal(frames,cframes) or not np.array_equal(frames,original_frames) or not np.allclose(matrices,original,atol=1e-10,rtol=0): raise ValueError('原始、转换相机及观测帧不一致')
    check_provenance(Path(source),derived)
    ply=Path(model)/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
    coeff=read_coefficients(ply,ids); valid=np.isfinite(y)&(w>0)
    if not valid.any(): raise ValueError('没有有效观测，无法计算 SH 截断率')
    result={'frames':frames,'gaussian_ids':ids}; diagnostics={}
    for L in (0,1,2):
        h,raw=sh_luminance(coeff,xyz,centers,L)
        if h.shape!=y.shape or raw.shape!=(*y.shape,3) or not np.isfinite(raw).all(): raise ValueError('SH 输出形状或数值无效')
        clipped=(raw<0)|(raw>1)
        fractions=clipped[valid].mean(axis=0); diagnostics[str(L)]={'low':(raw[valid]<0).mean(axis=0).tolist(),'high':(raw[valid]>1).mean(axis=0).tolist(),'invalid_clipping':bool((fractions>.2).any())}
        result[f'h_L{L}']=h; result[f'raw_L{L}']=raw
    np.savez_compressed(dest/'sh_values.npz',**result)
    (dest/'extraction.json').write_text(json.dumps({'input_sha256':digest(observations),'camera_sha256':digest(cf),'original_camera_sha256':digest(Path(source)/'camera.json'),'preparation_commands_sha256':digest(derived.parent/'preparation_commands.json'),'source_manifest_sha256':digest(derived/'rgb_only_manifest.json'),'ply_sha256':digest(ply),'gt_read':False,'diagnostics':diagnostics},indent=2))
    return diagnostics
