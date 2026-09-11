"""Run with normal Metal access. Small causal tests for the full-graph GPU kernel."""
import unittest,math
import numpy as np
from .full_engine import FullEngine

def graph():
    return dict(offsets=np.array([0,1,3,4,4],np.uint32),targets=np.array([1,2,3,3],np.uint32),counts=np.array([500,300,150,500],np.uint32),sign=np.array([1,1,-1,1],np.int32))
def reference(g,rates,steps=500,seed=7):
    n=len(rates);v=np.full(n,-52.,np.float32);conductance=np.zeros(n,np.float32);until=np.zeros(n,int);history={};windows=[];counts=np.zeros(n,np.float32)
    em=np.float32(math.exp(-.1/20));es=np.float32(math.exp(-.1/5));coupling=np.float32(5/15*(em-es))
    for tick in range(steps):
        current=np.zeros(n,np.int32)
        for source in history.get(tick-18,[]):
            for e in range(g['offsets'][source],g['offsets'][source+1]):
                target=g['targets'][e]
                if tick>=until[target]:current[target]+=int(g['counts'][e])*int(g['sign'][source])
        fired=[]
        for i in range(n):
            if tick<until[i]:continue
            v[i]=np.float32(-52+np.float32((v[i]+52)*em)+np.float32(conductance[i]*coupling));conductance[i]*=es
            spike=v[i]>-45
            conductance[i]+=np.float32(current[i]) * np.float32(.275)
            x=(((i+1)*747796405)&0xffffffff)^(((tick+1)*2891336453)&0xffffffff)^seed
            x=((x^(x>>16))*2246822519)&0xffffffff;x=((x^(x>>13))*3266489917)&0xffffffff;x=x^(x>>16)
            if np.float32(x)/np.float32(4294967296)<np.float32(rates[i]) * np.float32(.0001):v[i]+=np.float32(68.75)
            if spike:v[i]=-52;conductance[i]=0;until[i]=tick+22;counts[i]+=1;fired.append(i)
        history[tick]=fired
        if tick%100==99:windows.append(counts.copy());counts.fill(0)
    return windows

class SpikingTests(unittest.TestCase):
    def setUp(self):self.g=graph();self.engine=FullEngine(self.g)
    def tearDown(self):self.engine.close()
    def test_no_input_is_quiet(self):
        self.assertEqual(float(self.engine.batch(np.zeros(4)).sum()),0)
    def test_delay_refractory_and_inhibition_match_cpu(self):
        rates=np.array([400,0,0,0],np.float32);expected=reference(self.g,rates)
        for counts in expected:np.testing.assert_array_equal(self.engine.batch(rates),counts)
        self.assertGreater(sum(x[1] for x in expected),0)
    def test_recurrence_off_blocks_spread(self):
        rates=np.array([400,0,0,0],np.float32)
        counts=sum((self.engine.batch(rates,recurrent=False) for _ in range(5)),np.zeros(4))
        self.assertGreater(counts[0],0);self.assertEqual(float(counts[1:].sum()),0)
    def test_reset_replays_same_spikes(self):
        rates=np.array([400,0,0,0],np.float32)
        a=self.engine.batch(rates);self.engine.reset();b=self.engine.batch(rates)
        np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
