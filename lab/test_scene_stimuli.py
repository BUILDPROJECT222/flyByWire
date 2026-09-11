import unittest
import numpy as np
from .scene_stimuli import clip,image_motion
class SceneTests(unittest.TestCase):
    def test_labeled_motion_is_present(self):
        for seed in [2000,2001,2002]:
            for kind in ['texture','on_edge','off_edge']:
                for label in [-1,1]:
                    self.assertEqual(image_motion(clip(seed,kind,label)),label)
    def test_stationary_and_luminance_bounds(self):
        a=clip(2000,'stationary')
        self.assertEqual(image_motion(a),0)
        np.testing.assert_array_equal(a[0],a[-1])
        for kind in ['texture','on_edge','off_edge','looming']:
            a=clip(2000,kind)
            self.assertEqual(a.shape,(12,48,64))
            self.assertTrue(np.all((a>=0)&(a<=255)))
if __name__=='__main__':unittest.main()
