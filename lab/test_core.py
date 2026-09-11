import unittest
import numpy as np
from .core import Circuit,Arena,initial_weights,evaluate

class CircuitTests(unittest.TestCase):
    def test_graph_is_measured_and_consistent(self):
        c=Circuit()
        self.assertEqual(c.n,c.meta['manifest']['nodes'])
        self.assertEqual(len(c.source),c.meta['manifest']['edges'])
        self.assertEqual(int(c.contacts.sum()),c.meta['manifest']['contacts'])
        self.assertTrue(np.all(c.contacts>0))
        self.assertEqual(c.feature_size,81)
        self.assertEqual(set(c.bins),set(range(8)))

    def test_sensory_signal_propagates_and_is_bounded(self):
        dark=Circuit(); bright=Circuit()
        for _ in range(300):
            a=dark.step(np.zeros(8)); b=bright.step(np.ones(8))
        self.assertTrue(np.isfinite(b).all())
        self.assertTrue(np.all((bright.rates>=0)&(bright.rates<=1)))
        self.assertGreater(np.max(np.abs(a[:-1]-b[:-1])),.001)

    def test_disconnection_removes_input_from_readout(self):
        a=Circuit('disconnected'); b=Circuit('disconnected')
        for _ in range(100):
            fa=a.step(np.zeros(8)); fb=b.step(np.ones(8))
        np.testing.assert_array_equal(fa,fb)

    def test_shuffle_preserves_edges_and_weight_multiset(self):
        a=Circuit(); b=Circuit('shuffled')
        np.testing.assert_array_equal(a.W.indices,b.W.indices)
        np.testing.assert_array_equal(a.W.indptr,b.W.indptr)
        np.testing.assert_array_equal(np.sort(a.W.data),np.sort(b.W.data))
        self.assertFalse(np.array_equal(a.W.data,b.W.data))

    def test_seeded_evaluation_is_repeatable(self):
        c=Circuit(); w=initial_weights(c.feature_size)
        self.assertEqual(evaluate(c,w,seeds=(101,),steps=40),evaluate(c,w,seeds=(101,),steps=40))
        self.assertNotEqual(Arena(101).walls,Arena(102).walls)

if __name__=='__main__': unittest.main()
