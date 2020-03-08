import unittest

import fairyfishtest

class TestFairyFishTest(unittest.TestCase):
    def test_parse_time_control(self):
        tc = fairyfishtest.TimeControl.parse('10/20+3')
        self.assertEqual(tc.time, 20)
        self.assertEqual(tc.increment, 3)
        self.assertEqual(tc.moves, 10)

        tc = fairyfishtest.TimeControl.parse(str(tc))
        self.assertEqual(tc.time, 20)
        self.assertEqual(tc.increment, 3)
        self.assertEqual(tc.moves, 10)

        tc = fairyfishtest.TimeControl.parse('5.1+0.2')
        self.assertEqual(tc.time, 5.1)
        self.assertEqual(tc.increment, 0.2)
        self.assertEqual(tc.moves, 0)

        tc = fairyfishtest.TimeControl.parse('10')
        self.assertEqual(tc.time, 10)
        self.assertEqual(tc.increment, 0)
        self.assertEqual(tc.moves, 0)

    def test_xboard_time_control(self):
        tc = fairyfishtest.TimeControl.parse('10/20+3')
        self.assertEqual(tc.format_xboard(), '10 0:20 3')


if __name__ == '__main__':
    unittest.main(verbosity=2)
