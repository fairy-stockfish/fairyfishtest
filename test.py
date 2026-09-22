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
        self.assertEqual(fairyfishtest.Engine.move_to_uci('@@@@', True), '0000')

    def test_from_uci(self):
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', False), 'a1a2')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('h9h10+', True), 'h8h9+')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a10b10', True), 'a9b9')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('a1a2', True), 'a0a1')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('P@e6', True), 'P@e5')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('0000', True), '@@@@')
        self.assertEqual(fairyfishtest.Engine.move_from_uci('e1e1', True), '@@@@')


class TestBughouse(unittest.TestCase):
    def get_captured(self, start_fen, moves):
        game = fairyfishtest.Game.__new__(fairyfishtest.Game)
        game.variant = 'bughouse'
        return game.get_captured(start_fen, moves)

    def test_get_captured(self):
        self.assertEqual(self.get_captured('r2qk3/8/8/8/8/8/8/3RK3[] w - - 0 1', ['d1d8', 'e8d8']), 'R')
        # a promoted piece is passed on as a pawn
        self.assertEqual(self.get_captured('r3k3/1P6/8/8/8/8/8/4K2R[] w - - 0 1', ['b7b8q', 'a8b8']), 'P')
        self.assertEqual(self.get_captured('4k2r/8/8/8/8/8/1p6/R3K3[] b - - 0 1', ['b2b1q', 'a1b1']), 'p')
        # promoting does not pass on anything
        self.assertEqual(self.get_captured('4k3/1P6/8/8/8/8/8/4K2R[] w - - 0 1', ['b7b8q']), '')


class TestScoreboard(unittest.TestCase):
    def test_reserve_does_not_overshoot(self):
        scoreboard = fairyfishtest.Scoreboard(5)
        self.assertEqual(scoreboard.reserve(2), 2)
        self.assertEqual(scoreboard.reserve(2), 2)
        # only one game is left of the five
        self.assertEqual(scoreboard.reserve(2), 1)
        self.assertEqual(scoreboard.reserve(2), 0)

    def test_release(self):
        scoreboard = fairyfishtest.Scoreboard(2)
        self.assertEqual(scoreboard.reserve(2), 2)
        scoreboard.release(1)
        self.assertEqual(scoreboard.reserve(2), 1)

    def test_stop(self):
        scoreboard = fairyfishtest.Scoreboard(10)
        self.assertEqual(scoreboard.reserve(2), 2)
        scoreboard.stop()
        self.assertEqual(scoreboard.reserve(2), 0)
        # a release by a match that is shutting down does not revive the run
        scoreboard.release(2)
        self.assertEqual(scoreboard.reserve(2), 0)

    def test_record(self):
        scoreboard = fairyfishtest.Scoreboard(4)
        for result in (1, 1, -1, 0):
            scoreboard.record(result)
        self.assertEqual(scoreboard.score, [2, 1, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
