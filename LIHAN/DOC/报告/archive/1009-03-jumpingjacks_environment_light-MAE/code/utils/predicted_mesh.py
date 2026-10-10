"""模型重建三角网格的像素法线；不读取或拟合 GT 法线。"""
from pathlib import Path

import numpy as np

from utils.geometry_eval_utils import camera_rays, file_sha256, unit_normal


class PredictedMesh:
    def __init__(self, path, threads=4):
        import open3d as o3d
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        self.threads = threads
        mesh = o3d.io.read_triangle_mesh(str(self.path))
        if not mesh.has_triangles():
            raise ValueError('预测 Mesh 必须包含三角面；Gaussian PLY 或 GT 点云不能作为重建 Mesh。')
        self.vertices = np.asarray(mesh.vertices).astype(np.float32)
        self.triangles = np.asarray(mesh.triangles).astype(np.uint32)
        if not np.isfinite(self.vertices).all():
            raise ValueError('预测 Mesh 存在无效坐标。')
        # 总是从预测三角几何重新计算面积加权顶点法线，不信任输入 nx/ny/nz。
        mesh.compute_vertex_normals()
        self.normals = np.asarray(mesh.vertex_normals).copy()
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.triangles))

    def render(self, c2w, intrinsics, height, width):
        import open3d as o3d
        rays = camera_rays(c2w, intrinsics, height, width)
        hits = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=self.threads)
        depth = hits['t_hit'].numpy()
        valid = np.isfinite(depth) & (depth > 0)
        index = np.where(valid, hits['primitive_ids'].numpy(), 0).astype(np.int64)
        uv = hits['primitive_uvs'].numpy()
        weights = np.stack((1-uv.sum(-1), uv[..., 0], uv[..., 1]), -1)
        normal = (self.normals[self.triangles[index]] * weights[..., None]).sum(-2)
        normal, normal_valid = unit_normal(normal)
        # 与二维高斯双面渲染一致，朝向由相机射线决定，不按 GT 或误差选符号。
        flip = (normal * rays[..., 3:]).sum(-1) > 0
        normal[flip] *= -1
        valid &= normal_valid
        normal[~valid] = 0
        return dict(mesh_normal=normal, mesh_valid=valid,
                    mesh_depth=np.where(valid, depth, 0).astype(np.float32))
