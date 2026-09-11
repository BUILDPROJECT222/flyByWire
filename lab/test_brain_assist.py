import unittest
from .brain_assist import score_to_yaw
from .flight_runner import Sequencer, packet

class BrainAssistTests(unittest.TestCase):
    def test_score_to_yaw_deadzone_and_clamp(self):
        self.assertEqual(score_to_yaw(None), 0)
        self.assertEqual(score_to_yaw(0.01), 0)
        self.assertEqual(score_to_yaw(0.2), 1)
        self.assertEqual(score_to_yaw(10), 6)
        self.assertEqual(score_to_yaw(-10), -6)

    def test_sequencer_brain_assist_applies_yaw(self):
        s = Sequencer('brain-yaw-assist-v2')
        s.tick(0, ready=True)
        self.assertEqual(s.tick(1, ready=True, start=True), packet(1))
        self.assertEqual(s.tick(2), packet())
        self.assertEqual(s.tick(4, assist_yaw=4), packet(0, 3, 4))
        self.assertEqual(s.tick(4.1, assist_yaw=0), packet())
        self.assertEqual(s.tick(4.2, assist_yaw=99), packet(0, 3, 6))
        land = s.tick(10, assist_yaw=3)
        self.assertEqual(s.phase, 'landing')
        self.assertEqual(land, packet(2))

if __name__ == '__main__':
    unittest.main()
