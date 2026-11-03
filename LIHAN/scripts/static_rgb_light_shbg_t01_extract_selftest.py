"""合成数据检验真实提取循环的各阶维度和有效掩码。"""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest, extract


def main():
    T, G = 150, 5
    frames=np.arange(1,T+1); ids=np.arange(G)
    xyz=np.broadcast_to(np.array([1.,0.,0.]),(T,G,3)).copy()
    y=np.ones((T,G)); w=np.ones_like(y); w[::2,1::2]=0; y[1::3,0]=np.nan; w[1::3,0]=0
    valid=np.isfinite(y)&(w>0); centers=np.zeros((T,3)); matrices=np.broadcast_to(np.eye(4),(T,4,4)).copy()
    coeff=np.zeros((G,3,16)); coeff[[0,2],0,0]=3.0
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); data=root/'observations'/'extraction'; data.mkdir(parents=True); obs=data/'observations.npz'
        np.savez_compressed(obs,frames=frames,gaussian_ids=ids,frame_xyz=xyz,intensity=y,weights=w)
        derived=data.parent/'source_all'; derived.mkdir(); (derived/'transforms_all.json').write_text('{}'); (derived/'rgb_only_manifest.json').write_text('{}'); (data.parent/'preparation_commands.json').write_text('{}')
        source=root/'source'; source.mkdir(); (source/'camera.json').write_text('{}')
        model=root/'model'; ply=model/'point_cloud'/'iteration_35000'/'point_cloud.ply'; ply.parent.mkdir(parents=True); ply.write_bytes(b'synthetic')
        with patch('LIHAN.scripts.static_rgb_light_shbg_t01_extract.camera_fields',return_value=(frames,centers,matrices)), patch('LIHAN.scripts.static_rgb_light_shbg_t01_extract.raw_camera_fields',return_value=(frames,matrices)), patch('LIHAN.scripts.static_rgb_light_shbg_t01_extract.check_provenance'), patch('LIHAN.scripts.static_rgb_light_shbg_t01_extract.read_coefficients',return_value=coeff):
            diag=extract(obs,source,model,35000,root/'result',digest(obs),ids)
        with np.load(root/'result'/'sh_values.npz',allow_pickle=False) as result:
            for L in (0,1,2):
                h=result[f'h_L{L}']; raw=result[f'raw_L{L}']
                assert h.shape==(T,G) and raw.shape==(T,G,3)
                clipped=(raw<0)|(raw>1); fraction=clipped[valid].mean(axis=0)
                np.testing.assert_allclose(diag[str(L)]['high'],(raw[valid]>1).mean(axis=0))
                assert diag[str(L)]['invalid_clipping']==bool((fraction>.2).any()) and diag[str(L)]['invalid_clipping']
            np.testing.assert_allclose(result['raw_L0'],np.broadcast_to(result['raw_L0'][0:1],(T,G,3)),rtol=0,atol=0)
            np.testing.assert_allclose(result['h_L0'],np.broadcast_to(result['h_L0'][0:1],(T,G)),rtol=0,atol=0)
        saved=json.loads((root/'result'/'extraction.json').read_text()); assert saved['diagnostics']==diag and not saved['gt_read']
    print('PASS: 合成提取循环 L0/L1/L2 逐帧形状、有效掩码及截断率；无真实数据和 GT')

if __name__=='__main__': main()
