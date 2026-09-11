import unittest
import numpy as np
from scipy.sparse import csr_matrix
from .graded_engine import GradedEngine

class GradedTests(unittest.TestCase):
    def setUp(self):
        self.g=dict(offsets=np.array([0,0,1,3],np.uint32),sources=np.array([0,0,1],np.uint32),
                    weights=np.array([-1,.5,.5],np.float32))
        self.bias=np.array([0,.5,.5],np.float32)
        self.engine=GradedEngine(self.g,self.bias)
    def tearDown(self):self.engine.close()
    def test_cpu_parity_inhibition_and_reset(self):
        w=csr_matrix((self.g['weights'],self.g['sources'],self.g['offsets']),shape=(3,3))
        v=self.bias.copy();drive=np.array([1,0,0],np.float32)
        for _ in range(100):v+=np.float32(.1)*(-v+self.bias+np.float32(.8)*w.dot(np.maximum(v,0))+drive)
        actual=self.engine.batch(drive,100)
        np.testing.assert_allclose(actual,v,atol=1e-6)
        self.assertLess(actual[1],0)
        self.engine.reset();np.testing.assert_array_equal(actual,self.engine.batch(drive,100))
    def test_cell_parameters_match_cpu(self):
        bias=np.array([.1,.2,.3],np.float32);tau=np.array([10,50,100],np.float32)
        self.engine.configure(bias,tau);v=bias.copy();drive=np.array([1,0,0],np.float32)
        w=csr_matrix((self.g['weights'],self.g['sources'],self.g['offsets']),shape=(3,3))
        for _ in range(100):v+=(5/tau)*(-v+bias+.8*w.dot(np.maximum(v,0))+drive)
        np.testing.assert_allclose(self.engine.batch(drive,100),v,atol=1e-6)
        with self.assertRaises(ValueError):self.engine.configure(bias,np.array([0,50,100]))
    def test_warm_state_restore(self):
        drive=np.array([1,0,0],np.float32)
        warm=self.engine.batch(drive,25)
        expected=self.engine.batch(drive,4)
        self.engine.reset(warm)
        np.testing.assert_array_equal(expected,self.engine.batch(drive,4))
    def test_finite_bounded_and_bad_input(self):
        actual=self.engine.batch(np.ones(3,np.float32),1000)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertLess(np.max(np.abs(actual)),7.51)
        with self.assertRaises(ValueError):self.engine.batch(np.array([np.nan,0,0]))
if __name__=='__main__':unittest.main()
