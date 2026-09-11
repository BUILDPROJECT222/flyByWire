import tempfile,unittest
from pathlib import Path
import numpy as np
from .motion_readout import MotionReadout

class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'readout.npz'
        np.savez(self.path,train=np.array([[1,1,1],[-1,-1,-1]]),mean=np.zeros(3),
                 scale=np.ones(3),dual=np.array([1,-1]),indices=np.array([1]))
        self.model=MotionReadout(self.path)
    def tearDown(self):self.temp.cleanup()
    def test_history_gate_score_and_reset(self):
        for _ in range(11):self.assertIsNone(self.model.update(np.array([0,1])))
        result=self.model.update(np.array([0,1]))
        self.assertEqual(result['direction'],'right');self.assertAlmostEqual(result['score'],4)
        self.model.reset();self.assertIsNone(self.model.update(np.array([0,1])))
    def test_reject_nonfinite_state(self):
        with self.assertRaises(ValueError):self.model.update(np.array([0,np.nan]))

if __name__=='__main__':unittest.main()
