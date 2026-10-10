"""已知GT Alpha拒绝与原始RGBA/哈希逐图核验。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scene.pbr_directional import validate_directional_training_inputs


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class RGBAlphaProvenanceTest(unittest.TestCase):
    def fixture(self, root):
        (root/'images').mkdir()
        raw=root/'raw.png';target=root/'images/frame_0001.png'
        pixels=np.array([[[32,64,96,255],[100,200,50,128]],
                         [[50,60,70,0],[0,0,0,0]]],dtype=np.uint8)
        Image.fromarray(pixels,'RGBA').save(raw)
        converted=pixels.copy();converted[converted[...,3]==0,:3]=0
        Image.fromarray(converted,'RGBA').save(target)
        (root/'transforms_train.json').write_text(json.dumps(dict(frames=[dict(source_frame=1,file_path=target.name)])))
        (root/'transforms_test.json').write_text(json.dumps(dict(frames=[])))
        manifest=dict(summary=dict(mask_source='original-image-alpha'),frames={'0001':dict(
            source_image=str(raw),source_image_sha256=sha(raw),image_sha256=sha(target),split='train')})
        (root/'dataset_manifest.json').write_text(json.dumps(manifest))
        return raw,target,manifest

    def test_original_rgba_is_verified_without_material_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            audit=validate_directional_training_inputs(root)
            self.assertEqual(audit['status'],'PASS')
            self.assertEqual(audit['frames'],1)
            self.assertFalse(audit['material_geometry_GT_opened'])

    def test_known_albedo_alpha_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'dataset_manifest.json').write_text(json.dumps(dict(summary=dict(mask_source='albedo-alpha'))))
            with self.assertRaisesRegex(ValueError,'albedo Alpha'):
                validate_directional_training_inputs(root)

    def test_changed_source_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw,_,_=self.fixture(root)
            raw.write_bytes(raw.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'哈希'):
                validate_directional_training_inputs(root)

    def test_changed_alpha_is_rejected_even_with_updated_output_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,target,manifest=self.fixture(root)
            pixels=np.array(Image.open(target));pixels[0,1,3]=255
            Image.fromarray(pixels,'RGBA').save(target)
            manifest['frames']['0001']['image_sha256']=sha(target)
            (root/'dataset_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'与原始图像不一致'):
                validate_directional_training_inputs(root)
