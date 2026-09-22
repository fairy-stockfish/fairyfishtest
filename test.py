import logging
import sys
import unittest

import fairyfishtest
import stat_util


# A stand-in engine that stays alive until it is told to quit, without ever
# saying anything itself
IDLE_ENGINE = [sys.executable, '-c', 'import sys\nfor line in sys.stdin:\n'
               '    if line.strip() == "quit": break']
# The same, but it closes its output right away, like an engine that stops
# talking without dying
MUTE_ENGINE = [sys.executable, '-c', 'import os, sys\nos.close(1)\nfor line in sys.stdin:\n'
               '    if line.strip() == "quit": break']


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

        # a whole-second increment is announced exactly as before
        self.assertEqual(fairyfishtest.TimeControl.parse('60+2').format_xboard(), '0 1:0 2')
        self.assertEqual(fairyfishtest.TimeControl.parse('10+0').format_xboard(), '0 0:10 0')

        # a fractional increment needs an engine that reads it as a real number
        self.assertEqual(fairyfishtest.TimeControl.parse('10+0.1').format_xboard(), '0 0:10 0.1')
        self.assertEqual(fairyfishtest.TimeControl.parse('60+2.5').format_xboard(), '0 1:0 2.5')

        # --strict-xboard sticks to the whole seconds the protocol defines
        self.assertEqual(fairyfishtest.TimeControl.parse('10+0.1', True).format_xboard(), '0 0:10 0')
        self.assertEqual(fairyfishtest.TimeControl.parse('60+2.5', True).format_xboard(), '0 1:0 2')

        # a fractional base time is announced rounded down, since time and otim
        # override it with centisecond precision before every move anyway
        self.assertEqual(fairyfishtest.TimeControl.parse('10.5+1').format_xboard(), '0 0:10 1')

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


class TestEngineRestart(unittest.TestCase):
    """Covers what happens after an engine process died mid-run."""

    def make_engine(self, name='e1', args=IDLE_ENGINE):
        engine = fairyfishtest.Engine(args, {}, name)
        self.addCleanup(engine.close)
        return engine

    @staticmethod
    def kill(engine):
        engine.process.kill()
        engine.process.wait()

    def test_write_to_dead_engine_reports_eof(self):
        engine = self.make_engine()
        self.kill(engine)
        with self.assertLogs(level=logging.ERROR) as logs:
            # None of the write paths may raise a BrokenPipeError
            self.assertFalse(engine.go())
            self.assertFalse(engine.update_clocks(10, 10))
            self.assertFalse(engine.usermove('e2e4'))
            self.assertFalse(engine.holding('[] []'))
            self.assertFalse(engine.ptell('hi'))
            self.assertFalse(engine.newgame('chess', fairyfishtest.TimeControl(10)))
        # The game in progress sees the death like any other engine failure
        self.assertEqual(engine.get_move(timeout=1), ('eof', None))
        # and the log is not flooded with one error per command
        self.assertEqual(len(logs.output), 1)

    def test_restart_replaces_the_process(self):
        engine = self.make_engine()
        partner = self.make_engine('e2')
        engine.partner = partner
        old_process = engine.process
        self.kill(engine)
        self.assertFalse(engine.is_alive())
        with self.assertLogs(level=logging.WARNING):
            engine.restart()
        self.assertTrue(engine.is_alive())
        self.assertIsNot(engine.process, old_process)
        # the partner wiring of the other boards points at the Engine object
        self.assertIs(engine.partner, partner)
        # the fresh process accepts commands again, and its queue is clean
        self.assertTrue(engine.newgame('chess', fairyfishtest.TimeControl(10)))
        self.assertEqual(engine.get_move(timeout=0.1), ('timeout', None))

    def test_engine_that_closed_its_output_counts_as_dead(self):
        engine = self.make_engine(args=MUTE_ENGINE)
        self.assertEqual(engine.get_move(timeout=5), ('eof', None))
        # An engine that stopped talking is of no use even while it still runs
        self.assertIsNone(engine.process.poll())
        self.assertFalse(engine.is_alive())
        with self.assertLogs(level=logging.ERROR):
            self.assertFalse(engine.go())

    def make_match(self, engines, max_restarts):
        match = fairyfishtest.Match.__new__(fairyfishtest.Match)
        match.engines = engines
        match.board2_engines = None
        match.max_restarts = max_restarts
        match.restarts = 0
        return match

    def test_match_restarts_dead_engines(self):
        engines = [self.make_engine('e1'), self.make_engine('e2')]
        match = self.make_match(engines, 2)
        self.kill(engines[1])
        with self.assertLogs(level=logging.WARNING):
            match.restart_dead_engines()
        self.assertEqual(match.restarts, 1)
        self.assertTrue(all(engine.is_alive() for engine in engines))
        # a run without deaths does not consume the budget
        match.restart_dead_engines()
        self.assertEqual(match.restarts, 1)

    def test_match_gives_up_after_the_restart_limit(self):
        engine = self.make_engine()
        match = self.make_match([engine], 1)
        self.kill(engine)
        with self.assertLogs(level=logging.WARNING):
            match.restart_dead_engines()
        self.kill(engine)
        with self.assertRaises(RuntimeError):
            match.restart_dead_engines()


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


class TestStatistics(unittest.TestCase):
    def test_elo(self):
        elo, elo95, los = stat_util.get_elo([63, 34, 3])
        self.assertAlmostEqual(elo, 103.73, places=2)
        self.assertAlmostEqual(elo95, 71.10, places=2)
        self.assertAlmostEqual(los, 0.9990, places=4)

        # a balanced score is no difference at all, with an even chance either way
        elo, _, los = stat_util.get_elo([50, 50, 0])
        self.assertAlmostEqual(elo, 0, places=6)
        self.assertAlmostEqual(los, 0.5, places=6)

    def test_elo_undefined(self):
        # the bounds run out of the defined range before the score has spread
        self.assertEqual(fairyfishtest.format_elo([2, 1, 1]), '')
        # one bound past the defined range leaves an estimate but no interval
        self.assertIn('95% interval undefined', fairyfishtest.format_elo([0, 2, 1]))
        self.assertIn('+-71.1 (95%)', fairyfishtest.format_elo([63, 34, 3]))

    def test_sprt(self):
        # not enough of a spread to estimate drawelo yet
        self.assertEqual(fairyfishtest.sprt([10, 0, 20], 0, 5)['llr'], 0.0)

        undecided = fairyfishtest.sprt([716, 591, 2163], 0, 5)
        self.assertAlmostEqual(undecided['llr'], 2.5669, places=4)
        self.assertFalse(undecided['finished'])

        self.assertEqual(fairyfishtest.sprt([600, 400, 1000], 0, 5)['state'], 'accepted')
        self.assertEqual(fairyfishtest.sprt([400, 600, 1000], 0, 5)['state'], 'rejected')

    def test_scoreboard_stops_on_sprt(self):
        scoreboard = fairyfishtest.Scoreboard(10000, (0, 5))
        # A clearly stronger engine 1, so the test accepts H1 well before the
        # game limit. The results have to cover wins, losses and draws alike,
        # since the test cannot estimate drawelo out of a sample without them.
        results = [1, 1, 0, -1]
        played = 0
        while scoreboard.reserve(1):
            scoreboard.record(results[played % len(results)])
            played += 1
        self.assertLess(played, 10000)
        self.assertEqual(fairyfishtest.sprt(scoreboard.score, 0, 5)['state'], 'accepted')


if __name__ == '__main__':
    unittest.main(verbosity=2)
