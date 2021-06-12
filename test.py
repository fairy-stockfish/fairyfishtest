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

    def test_to_uci(self):
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a1a2', False), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('h8h9+', True), 'h9h10+')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a9b9', True), 'a10b10')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('a0a1', True), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_to_uci('P@e5', True), 'P@e6')

    def test_from_uci(self):
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', False), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('h9h10+', True), 'h8h9+')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a10b10', True), 'a9b9')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', True), 'a0a1')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('P@e6', True), 'P@e5')


if __name__ == '__main__':
    unittest.main(verbosity=2)
