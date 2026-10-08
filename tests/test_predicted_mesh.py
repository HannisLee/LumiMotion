"""已知平面验证 mesh 射线、世界坐标、相机 Z 和法线方向。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from utils.predicted_mesh import PredictedMesh


class PredictedMeshTest(unittest.TestCase):
    def plane(self, directory):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[-1,-1,-2], [1,-1,-2], [1,1,-2], [-1,1,-2]])
        mesh.triangles = o3d.utility.Vector3iVector([[0,1,2], [0,2,3]])
        path = Path(directory)/'plane.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        return path

    def test_plane_first_hit_normal_z_and_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = PredictedMesh(self.plane(directory))
            output = renderer.render(np.eye(4), [5,5,3.5,3.5], 7, 7)
            mask = output['mesh_valid']
            self.assertTrue(mask[3,3])
            self.assertFalse(mask[0,0])
            np.testing.assert_allclose(output['mesh_depth'][mask], 2, atol=1e-6)
            np.testing.assert_allclose(output['mesh_normal'][mask],
                np.tile([0,0,1.], (int(mask.sum()),1)), atol=1e-6)

    def test_plane_world_translation_uses_same_camera_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = PredictedMesh(self.plane(directory))
            camera = np.eye(4); camera[2,3] = 3
            output = renderer.render(camera, [5,5,.5,.5], 1, 1)
            self.assertTrue(output['mesh_valid'][0,0])
            self.assertAlmostEqual(output['mesh_depth'][0,0], 5, delta=1e-6)
            np.testing.assert_allclose(output['mesh_normal'][0,0], [0,0,1], atol=1e-6)

    def test_point_cloud_cannot_be_used_as_predicted_triangle_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector([[0,0,-2]])
            path = Path(directory)/'points.ply'
            o3d.io.write_point_cloud(str(path), cloud)
            with self.assertRaises(ValueError):
                PredictedMesh(path)

    def test_rotated_camera_normal_remains_in_world_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector([[-2,-1,-1],[-2,1,-1],[-2,1,1],[-2,-1,1]])
            mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
            path = Path(directory)/'plane_x.ply'
            o3d.io.write_triangle_mesh(str(path),mesh)
            camera = np.eye(4)
            camera[:3,:3] = [[0,0,1],[0,1,0],[-1,0,0]]
            output = PredictedMesh(path).render(camera,[5,5,.5,.5],1,1)
            self.assertTrue(output['mesh_valid'][0,0])
            self.assertAlmostEqual(output['mesh_depth'][0,0],2,delta=1e-6)
            np.testing.assert_allclose(output['mesh_normal'][0,0],[1,0,0],atol=1e-6)


if __name__ == '__main__':
    unittest.main()
