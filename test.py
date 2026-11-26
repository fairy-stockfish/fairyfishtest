import json
import logging
import os
import re
import random
import sys
import tempfile
import unittest
import unittest.mock

import pyffish as sf

import combine_results
import fairyfishtest
import make_book
import stat_util


# A stand-in engine that stays alive until it is told to quit, without ever
# saying anything itself
IDLE_ENGINE = [sys.executable, '-c', 'import sys\nfor line in sys.stdin:\n'
               '    if line.strip() == "quit": break']
# The same, but it closes its output right away, like an engine that stops
# talking without dying
MUTE_ENGINE = [sys.executable, '-c', 'import os, sys\nos.close(1)\nfor line in sys.stdin:\n'
               '    if line.strip() == "quit": break']
# A UCI engine that evaluates every position as the score given by its first
# argument, in a stand-in for an engine that can search
UCI_ENGINE = '''import sys
for line in sys.stdin:
    line = line.strip()
    if line == "uci":
        print("uciok")
    elif line == "isready":
        print("readyok")
    elif line.startswith("go"):
        print("info depth 1 score cp " + sys.argv[1])
        print("bestmove 0000")
    elif line == "quit":
        break
    sys.stdout.flush()
'''

# An xboard engine that plays random legal moves, for matches without a real engine
XBOARD_ENGINE = '''import random, sys
import pyffish as sf
variant, fen, moves, playing = "chess", None, [], False
for line in sys.stdin:
    command = line.split()
    if not command:
        continue
    if command[0] in ("new", "force"):
        playing = False
        if command[0] == "new":
            fen, moves = None, []
    elif command[0] == "variant":
        variant = command[1]
    elif command[0] == "setboard":
        fen = line.split(None, 1)[1].strip()
    elif command[0] in ("usermove", "go"):
        if command[0] == "usermove":
            moves.append(command[1])
        else:
            playing = True
        start = fen or sf.start_fen(variant)
        legal = sf.legal_moves(variant, start, moves) if playing else []
        if legal:
            moves.append(random.choice(legal))
            print("move " + moves[-1])
            sys.stdout.flush()
    elif command[0] == "quit":
        break
'''


def write_script(directory, name, source):
    path = os.path.join(directory, name)
    with open(path, 'w') as script:
        script.write('#!{}\n{}'.format(sys.executable, source))
    os.chmod(path, 0o755)
    return path


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


class TestMatchOutput(unittest.TestCase):
    """Plays real matches between random movers, and checks what they write."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.engine = write_script(self.directory, 'random-engine', XBOARD_ENGINE)

    def run_match(self, games, variant='micro', **kwargs):
        pgn = os.path.join(self.directory, 'games.pgn')
        result = os.path.join(self.directory, 'result.json')
        with self.assertLogs(level=logging.INFO):
            fairyfishtest.main(self.engine, self.engine, {}, {}, fairyfishtest.TimeControl(10), variant,
                               games, 1, None, False, 0, 10, pgn_file=pgn, json_file=result, **kwargs)
        with open(pgn) as pgn_file, open(result) as result_file:
            return pgn_file.read(), json.load(result_file)

    def test_pgn_and_json(self):
        pgn, result = self.run_match(4)
        self.assertEqual(result['games'], 4)
        self.assertEqual(result['wins'] + result['losses'] + result['draws'], 4)
        # two pairs of games with swapped colors
        self.assertEqual(result['pairs'], 2)
        self.assertEqual(sum(result['pentanomial']), 2)
        self.assertEqual(result['engine1'], 'random-engine (1)')
        self.assertEqual(result['time_losses'], [0, 0])
        self.assertEqual(result['crashes'], [0, 0])
        self.assertEqual(result['variant'], 'micro')
        self.assertEqual(pgn.count('[Event '), 4)
        self.assertEqual(pgn.count('[Variant "micro"]'), 4)
        self.assertEqual(pgn.count('[Pair '), 0)
        # the engines swap colors from one game of a pair to the next
        self.assertIn('[White "random-engine (1)"]', pgn)
        self.assertIn('[White "random-engine (2)"]', pgn)
        self.assertIn('[%clk ', pgn)

    def test_pgn_moves_replay(self):
        pgn, _ = self.run_match(2)
        for game in pgn.split('[Event ')[1:]:
            moves = game.split('\n\n')[1].replace('\n', ' ')
            self.assertTrue(moves.startswith('1. '))
            self.assertTrue(moves.strip().endswith(('1-0', '0-1', '1/2-1/2', '*')))

    def test_two_board_pgn_replays_with_transfers(self):
        written = []
        original = fairyfishtest.PgnWriter.format_game

        def spy(writer, game, number, board=None):
            written.append(game)
            return original(writer, game, number, board)

        # random games of bughouse can go on for very long, so this engine resigns early
        source = XBOARD_ENGINE.replace('        legal = sf.legal', '        if len(moves) > 60:\n'
                                       '            print("resign")\n            sys.stdout.flush()\n'
                                       '            continue\n        legal = sf.legal', 1)
        self.engine = write_script(self.directory, 'resigning-engine', source)
        with unittest.mock.patch.object(fairyfishtest.PgnWriter, 'format_game', spy):
            pgn, _ = self.run_match(2, variant='bughouse')
        games = pgn.split('[Event ')[1:]
        self.assertEqual(len(games), 4)
        self.assertEqual(len(written), 4)
        transfers = 0
        for text, game in zip(games, written):
            header, movetext = text.split('\n\n')[:2]
            fen = re.search(r'\[FEN "([^"]*)"\]', header)
            fen = fen.group(1) if fen else sf.start_fen('bughouse')
            # the start position has the holdings of the start, not the received ones
            self.assertEqual(fen, game.start_fen)
            moves = []
            for token in re.findall(r'\{[^}]*\}|\S+', movetext):
                held = re.fullmatch(r'\{\[%holding \+(\w+)\]\}', token)
                if held:
                    transfers += 1
                    board, rest = sf.get_fen('bughouse', fen, moves).split('[', 1)
                    fen, moves = board + '[' + rest.split(']', 1)[0] + held.group(1) + ']' + rest.split(']', 1)[1], []
                elif not token.startswith('{') and not re.fullmatch(r'\d+\.+|1-0|0-1|1/2-1/2|\*', token):
                    # the SAN move has to be legal with the pieces received so far
                    legal = [m for m in sf.legal_moves('bughouse', fen, moves)
                             if sf.get_san('bughouse', sf.get_fen('bughouse', fen, moves), m) == token]
                    self.assertEqual(len(legal), 1, token)
                    moves.append(legal[0])
            replayed = sf.get_fen('bughouse', fen, moves)
            expected = sf.get_fen('bughouse', game.get_start_fen(), game.moves)
            self.assertEqual(replayed.split()[0:2], expected.split()[0:2])
            self.assertEqual(sorted(replayed.split('[')[1].split(']')[0]),
                             sorted(expected.split('[')[1].split(']')[0]))
        self.assertGreater(transfers, 0)

    def test_result_tag(self):
        game = fairyfishtest.Game.__new__(fairyfishtest.Game)
        game.start_fen = 'k7/8/8/8/8/8/8/K7 b - - 0 1'
        game.partner = None
        for result, tag in ((1, '0-1'), (-1, '1-0'), (0, '1/2-1/2'), (None, '*')):
            game.result = result
            self.assertEqual(fairyfishtest.PgnWriter.result_tag(game), tag)

    def test_engine_names(self):
        self.assertEqual(fairyfishtest.engine_names('/a/sf', '/b/sf'), ['sf (1)', 'sf (2)'])
        self.assertEqual(fairyfishtest.engine_names('/a/new', '/b/old'), ['new', 'old'])
        self.assertEqual(fairyfishtest.engine_names('/a/sf', '/b/sf', 'A', 'B'), ['A', 'B'])


class TestFailureStatistics(unittest.TestCase):
    def test_record_failures(self):
        scoreboard = fairyfishtest.Scoreboard(4)
        scoreboard.record(-1, [(1, 'time')])
        scoreboard.record(1, [(2, 'crash'), (2, 'time')])
        result = scoreboard.to_dict()
        self.assertEqual(result['time_losses'], [1, 1])
        self.assertEqual(result['crashes'], [0, 1])
        self.assertEqual(result['illegal_moves'], [0, 0])
        self.assertEqual((result['wins'], result['losses'], result['draws']), (1, 1, 0))

    def test_pentanomial(self):
        scoreboard = fairyfishtest.Scoreboard(8)
        for first, second in ((-1, -1), (-1, 0), (0, 0), (1, -1), (0, 1), (1, 1), (1, 1)):
            scoreboard.record_pair(first, second)
        self.assertEqual(scoreboard.to_dict()['pentanomial'], [1, 1, 2, 1, 2])

    def test_game_failure_of_the_engine_that_lost_on_time(self):
        game = fairyfishtest.Game(None, None, fairyfishtest.TimeControl(10), 'chess', None, (2, 1))
        game.moves = ['e2e4']
        game.clock_times = [-1, 10]
        self.assertTrue(game.is_game_end())
        self.assertEqual(game.failure, (0, 'time'))
        self.assertEqual(game.engine_ids[game.failure[0]], 2)


class TestCombineResults(unittest.TestCase):
    RESULT = {'engine1': 'a', 'engine2': 'b', 'variant': 'chess', 'time_control': '10+0', 'book': None,
              'wins': 10, 'losses': 5, 'draws': 5, 'pairs': 10, 'pentanomial': [1, 2, 3, 2, 2],
              'time_losses': [1, 0], 'crashes': [0, 2], 'illegal_moves': [0, 0], 'engine_restarts': 1}

    def test_combine(self):
        total = combine_results.combine([self.RESULT, self.RESULT])
        self.assertEqual((total['wins'], total['losses'], total['draws'], total['games']), (20, 10, 10, 40))
        self.assertEqual(total['pentanomial'], [2, 4, 6, 4, 4])
        self.assertEqual(total['time_losses'], [2, 0])
        self.assertEqual(total['crashes'], [0, 4])
        self.assertEqual(total['engine_restarts'], 2)

    def test_mismatches(self):
        self.assertEqual(combine_results.mismatches([self.RESULT, self.RESULT]), [])
        other = dict(self.RESULT, variant='crazyhouse')
        self.assertEqual(combine_results.mismatches([self.RESULT, other]), ['variant: chess, crazyhouse'])

    def test_report(self):
        total = combine_results.combine([self.RESULT, self.RESULT])
        lines = combine_results.report(total, 0, 5)
        self.assertIn('Total: 40 W: 20 L: 10 D: 10', lines[0])
        self.assertTrue(lines[1].startswith('Pairs: 20 [2, 4, 6, 4, 4] ELO: '))
        self.assertIn('Time losses: [2, 0]', lines[2])
        self.assertTrue(lines[3].startswith('SPRT [0.00,5.00]: LLR: '))

    def test_pentanomial_elo(self):
        elo, _, los = stat_util.get_elo_pentanomial([1, 2, 6, 2, 1])
        self.assertAlmostEqual(elo, 0, places=6)
        self.assertAlmostEqual(los, 0.5, places=6)
        elo, elo95, los = stat_util.get_elo_pentanomial([1, 2, 3, 6, 8])
        self.assertGreater(elo, 0)
        self.assertGreater(los, 0.95)


class TestMakeBook(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def test_random_opening(self):
        rng = random.Random(1)
        for variant in ('chess', 'crazyhouse', 'bughouse'):
            fens = [make_book.random_opening(variant, 6, rng) for _ in range(20)]
            fens = [fen for fen in fens if fen]
            self.assertTrue(fens)
            for fen in fens:
                self.assertEqual(sf.validate_fen(fen, variant), sf.FEN_OK)
                self.assertTrue(sf.legal_moves(variant, fen, []))

    def test_bughouse_holdings_hold_the_captures(self):
        rng = random.Random(3)
        fens = [make_book.random_opening('bughouse', 10, rng) for _ in range(200)]
        fens = [fen for fen in fens if fen]
        self.assertTrue(all(']' in fen for fen in fens))
        self.assertTrue(any(not fen.split('[')[1].startswith(']') for fen in fens))

    def test_generate_filters_by_score(self):
        def factory(score):
            path = write_script(self.directory, 'uci-{}'.format(score), UCI_ENGINE)
            wrapper = write_script(self.directory, 'run-{}'.format(score),
                                   'import os, sys\nos.execv({!r}, [{!r}, {!r}])\n'.format(path, path, str(score)))
            return lambda: make_book.UciEngine(wrapper, 'chess')

        book, _ = make_book.generate(factory(30), 'chess', 5, 4, 6, 100, 'go depth 1', random.Random(0))
        self.assertEqual(len(book), 5)
        self.assertEqual(len(set(book)), 5)
        # an engine that sees every position as lost leaves nothing to keep
        book, attempts = make_book.generate(factory(500), 'chess', 5, 4, 6, 100, 'go depth 1',
                                            random.Random(0), max_attempts=20)
        self.assertEqual(book, [])
        self.assertEqual(attempts, 20)


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
