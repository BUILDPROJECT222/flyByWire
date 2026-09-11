import unittest
import numpy as np
from .full_vision import RetinaEncoder

class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.encoder=RetinaEncoder(dict(nodes=5,receptors=[
            dict(index=1,uv=[0,0]),dict(index=3,uv=[1,1]),dict(index=4,uv=[.5,.5])]))
    def test_black_white_and_unmapped(self):
        np.testing.assert_array_equal(self.encoder.encode(np.zeros((2,2))),np.zeros(5))
        np.testing.assert_allclose(self.encoder.encode(np.full((2,2),255)),[0,120,0,120,120])
    def test_spatial_sampling_and_fixed_scale(self):
        np.testing.assert_allclose(self.encoder.encode(np.array([[0,0],[255,255]])),[0,0,0,120,60])
    def test_rgb_luminance(self):
        np.testing.assert_allclose(self.encoder.encode(np.full((2,2,3),[255,0,0])),[0,25.512,0,25.512,25.512],rtol=1e-5)
    def test_invalid_frame_rejected(self):
        for a in [np.array([[0,np.nan],[0,0]]),np.full((2,2),256),np.zeros((1,2))]:
            with self.assertRaises(ValueError):self.encoder.encode(a)

if __name__=='__main__':unittest.main()
