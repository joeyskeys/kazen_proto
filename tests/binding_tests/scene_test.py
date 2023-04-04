import unittest

import sys, os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, '../../build/src/binding'))
import pyzen

class TestSceneMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = pyzen.api.Scene()
        return super().setUp()

    def test_add_mesh(self):
        mat = pyzen.mat.Mat4f()
        verts = [pyzen.vec.Vec3f(0, 1, 2)]
        normals = [pyzen.vec.Vec3f(0, 1, 2)]
        uvs = [pyzen.vec.Vec2f(0, 1)]
        faces = [pyzen.vec.Vec3i(0, 1, 2)]
        self.assertEqual(self.scene.add_mesh(mat, verts, normals, uvs, faces, 'test', False), None)


if __name__ == '__main__':
    unittest.main()