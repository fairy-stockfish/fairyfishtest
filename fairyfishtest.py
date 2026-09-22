#!/usr/bin/env python3
#
# fairyfishtest is a script for automated chess variant engine matches.
# Copyright (C) 2020 Fabian Fichter
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import argparse
from collections import Counter
import contextlib
import logging
import math
import os
import queue
import random
import subprocess
import threading
import time

import pyffish as sf

import stat_util


# Grace period on top of the remaining clock before an unresponsive engine is
# counted as having lost on time.
MOVE_TIMEOUT_GRACE = 10

# Error rates and the drawelo the SPRT is run with, as in fishtest.
SPRT_ALPHA = SPRT_BETA = 0.05
SPRT_DRAWELO = 200

# How often a match replaces the process of an engine that died before it gives
# up, so that a reproducibly crashing engine does not restart forever.
MAX_ENGINE_RESTARTS = 10


class Engine:
    def __init__(self, args, options, name=None):
        self.args = args
        self.name = name
        self.lock = threading.Lock()
        self.partner = None
        self.rank_conversion = False  # convert ranks from zero-based to one-based
        self.options = options
        self._start()

    def _start(self):
        """Spawns the engine process and the thread draining its output."""
        self.process = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        universal_newlines=True)
        # The engine's output is drained by a dedicated thread, so that partner
        # tells are relayed as soon as they appear rather than only while we
        # happen to be waiting for this engine's move, and so that the pipe
        # never fills up while the engine is idle. A restart gets a fresh queue,
        # so that the EOF of the previous process is not mistaken for this one's.
        self.replies = queue.Queue()
        self.eof = False
        self.death_reported = False
        self.reader = threading.Thread(target=self._read_loop, args=(self.process,), daemon=True,
                                       name='{}:reader'.format(self.name) if self.name else None)
        self.reader.start()

    def is_alive(self):
        """Whether the engine still runs and still talks to us.

        An engine that closed its output is of no use to us either, even if its
        process happens to linger, so it counts as dead and gets restarted.
        """
        return self.process.poll() is None and not self.eof

    def restart(self):
        """Replaces a dead engine process by a freshly initialized one.

        The Engine object itself is kept, so that the partner wiring of the
        other boards keeps pointing at this engine.
        """
        logging.warning('Restarting engine {}.'.format(self.name))
        self.close()
        self._start()
        self.initialize()

    def _write(self, *lines):
        """Sends commands to the engine. Returns False if it is not listening.

        A dead engine is reported through the reply queue rather than by raising,
        so that the game in progress ends like any other engine failure instead
        of tearing down the whole run with a BrokenPipeError.
        """
        with self.lock:
            returncode = self.process.poll()
            if returncode is not None:
                self._report_death('the process exited with {}'.format(returncode))
                return False
            if self.eof:
                self._report_death('the process closed its output')
                return False
            try:
                for line in lines:
                    self.process.stdin.write(line + '\n')
                self.process.stdin.flush()
            except (OSError, ValueError) as e:
                self._report_death(e)
                return False
            return True

    def _report_death(self, reason):
        # Only the first failure is worth an error, the rest of the game keeps
        # talking to the same dead engine and would just flood the log.
        message = 'Engine {} is not accepting commands: {}'.format(self.name, reason)
        if self.death_reported:
            logging.debug(message)
        else:
            logging.error(message)
            self.death_reported = True
        self.replies.put(('eof', None))

    def initialize(self):
        options = ['option {}={}'.format(option, value) for option, value in self.options.items()]
        return self._write('xboard', 'protover 2', *options)

    def newgame(self, variant, time_control, start_fen=None):
        # Drop anything the engine may still have said about the previous game,
        # e.g. a move it finished after the game had already been decided.
        while True:
            try:
                stale = self.replies.get_nowait()
            except queue.Empty:
                break
            logging.debug('Discarding stale reply: {}'.format(stale))
        self.rank_conversion = sf.start_fen(variant).count('/') + 1 == 10
        return self._write(
            # Make sure the engine is not still searching when 'new' arrives
            'force',
            'new',
            'variant {}'.format(variant),
            # Park the engine in force mode before setting up the board. 'new'
            # assigned it a color, and setboard makes it start playing right
            # away when that color is the one to move, which happens as soon as
            # an opening book hands out a position with black to move. Each
            # engine takes over its color when the game sends it 'go'.
            'force',
            *(['setboard {}'.format(start_fen)] if start_fen else []),
            *(['partner test'] if self.partner else []),
            'level {}'.format(time_control.format_xboard()))

    def update_clocks(self, time, otim):
        # times in centiseconds
        return self._write('time {}'.format(int(time * 100)), 'otim {}'.format(int(otim * 100)))

    def go(self):
        return self._write('go')

    def _read_loop(self, process):
        """Drains the engine's output and relays partner tells immediately.

        Runs for the lifetime of the engine. Moves and claimed results are
        handed to get_move() through a queue, everything else is logged.
        """
        try:
            for line in process.stdout:
                partner = self.partner
                if partner and line.startswith('tellics ptell'):
                    fields = line.strip().split(None, 2)
                    if len(fields) > 2:
                        logging.debug('ptell relayed: {}'.format(fields[2]))
                        partner.ptell(fields[2])
                elif line.startswith('move'):
                    fields = line.strip().split()
                    if len(fields) > 1:
                        self.replies.put(('move', fields[1]))
                    else:
                        logging.error('Engine sent a move without a move: {}'.format(line.strip()))
                elif line.startswith(('1-0', '0-1', '1/2-1/2')):
                    self.replies.put(('result', line.strip().split()[0]))
                elif line.startswith('resign'):
                    self.replies.put(('resign', None))
                elif line.startswith(('Illegal move', 'Error')):
                    logging.error('Engine rejected a command: {}'.format(line.strip()))
        except (OSError, ValueError):
            pass  # pipe closed while we were reading
        finally:
            if self.process is process:  # not a leftover reader of a restarted engine
                self.eof = True
            self.replies.put(('eof', None))

    def get_move(self, timeout=None):
        """Returns (kind, value) where kind is move, result, resign, eof or timeout."""
        try:
            kind, value = self.replies.get(timeout=timeout)
        except queue.Empty:
            return 'timeout', None
        if kind == 'eof':
            # Keep reporting EOF rather than blocking forever on a dead engine
            self.replies.put(('eof', None))
        elif kind == 'move':
            return kind, self.move_to_uci(value, self.rank_conversion)
        return kind, value

    def close(self):
        try:
            if self.process.poll() is None:
                with self.lock:
                    self.process.stdin.write('quit\n')
                    self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        for stop in (self.process.terminate, self.process.kill):
            try:
                self.process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                stop()
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self.reader.join(timeout=5)

    @staticmethod
    def move_to_uci(move, rank_conversion):
        if move == '@@@@':
            return '0000'
        elif rank_conversion and move:
            square1 = move[0] + str(int(move[1]) + 1) if move[1].isnumeric() else move[0:2]
            square2 = move[2] + str(int(move[3]) + 1) if move[3].isnumeric() else move[2:4]
            return square1 + square2 + move[4:]
        else:
            return move

    @staticmethod
    def move_from_uci(move, rank_conversion):
        if len(move) % 2 == 0 and move[:len(move) // 2] == move[len(move) // 2:]:
            return '@@@@'
        elif rank_conversion and move and len(move) >= 4:
            len1 = 2 + move[2].isnumeric()
            len2 = 2 + (len(move) > len1 + 2 and move[len1 + 2].isnumeric())
            square1 = move[0] + str(int(move[1:len1]) - 1) if move[1:len1].isnumeric() else move[0:len1]
            square2 = (move[len1] + str(int(move[len1 + 1:len1 + len2]) - 1)
                       if move[len1 + 1:len1 + len2].isnumeric() else move[len1 + 1:len1 + len2])
            return square1 + square2 + move[len1 + len2:]
        else:
            return move

    def ptell(self, message):
        return self._write('ptell {}'.format(message))

    def holding(self, holding):
        return self._write('holding {}'.format(holding))

    def usermove(self, move, activate=False):
        """Sends the opponent's move, and with activate the go that answers it.

        The two go together in one write, so that the engine cannot be left
        sitting in force mode with a move it was never asked to answer.
        """
        return self._write('usermove {}'.format(self.move_from_uci(move, self.rank_conversion)),
                           *(['go'] if activate else []))


class TimeControl:
    # A base time below a second is announced as "0:0", which leaves the engine
    # without a time limit: Fairy-Stockfish discards the time and otim updates
    # unless the level command announced a non-zero base time, and then never
    # returns a move.
    MIN_BASE_TIME = 1

    def __init__(self, time, increment=0, moves=0, strict=False):
        self.time = time
        self.increment = increment
        self.moves = moves
        self.strict = strict

    @staticmethod
    def parse(stringified_tc, strict=False):
        time_and_increment = stringified_tc.split('+')
        moves_and_time = time_and_increment[0].split('/')
        time = float(moves_and_time[-1])
        increment = float(time_and_increment[1]) if len(time_and_increment) > 1 else 0
        moves = int(moves_and_time[0]) if len(moves_and_time) > 1 else 0
        return TimeControl(time, increment, moves, strict)

    def warn_if_inexact(self):
        """Warns about the part of the time control the level command cannot convey.

        The base time needs no warning, however odd it looks in the level
        command: time and otim carry the clock in centiseconds before every
        move and override whatever level announced. The increment has no such
        channel, and the protocol only defines it in whole seconds.
        """
        if self.increment == int(self.increment):
            return
        if self.strict:
            logging.warning('Increment {} is not a whole number of seconds, and --strict-xboard is set, '
                            'so it is announced as {} instead. The engine budgets its time for the '
                            'announced increment while the clocks are kept for the requested one.'
                            .format(self.increment, int(self.increment)))
        else:
            logging.warning('Increment {} is not a whole number of seconds, which the xboard level '
                            'command does not define. It is announced as such anyway, which needs an '
                            'engine that reads the increment as a real number, such as Fairy-Stockfish. '
                            'Other engines read an increment of {} instead; pass --strict-xboard to '
                            'announce that to every engine.'
                            .format(self.increment, int(self.increment)))

    def format_increment(self):
        """Formats the increment for the level command.

        The protocol defines whole seconds, which --strict-xboard sticks to.
        Otherwise a fractional increment is announced at the millisecond
        resolution the engines keep their clocks in, with whole numbers still
        formatted as whole numbers so that a conventional time control is
        announced exactly as before.
        """
        if self.strict:
            return str(int(self.increment))
        return '{:.3f}'.format(self.increment).rstrip('0').rstrip('.')

    def format_xboard(self):
        return '{} {}:{} {}'.format(self.moves, int(self.time // 60), int(self.time % 60),
                                    self.format_increment())

    def __str__(self):
        return ('{}/'.format(self.moves) if self.moves else '') + '{}+{}'.format(self.time, self.increment)


class Game:
    def __init__(self, engine1, engine2, time_control, variant='chess', start_fen=None):
        self.engines = [engine1, engine2]
        self.time_control = time_control
        self.variant = variant
        self.start_fen = start_fen or sf.start_fen(variant)
        self.moves = []
        self.result = None
        self.partner = None
        self.obtained_holdings = ''
        self.clock_times = [self.time_control.time, self.time_control.time]
        self.lock = threading.RLock()

    def initialize(self):
        start_fen = self.get_start_fen()
        # Only set up the board when the game does not start from the variant's
        # own start position, so that a run without an opening book talks to
        # the engines exactly as it did before.
        if start_fen == sf.start_fen(self.variant):
            start_fen = None
        for engine in self.engines:
            engine.newgame(self.variant, self.time_control, start_fen)

    def is_legal(self):
        assert self.moves
        return self.moves[-1] in sf.legal_moves(self.variant, self.get_start_fen(), self.moves[:-1])

    def is_game_end(self):
        with self.lock:
            game_end = False
            if self.moves and self.clock_times[(len(self.moves) - 1) % 2] <= 0:
                # time loss
                logging.warning('Engine {} loses on time.'.format((len(self.moves) - 1) % 2 + 1))
                result = 1
                game_end = True
            elif self.moves and not self.is_legal():
                # last move was illegal
                result = 1
                game_end = True
                logging.error('Illegal move: {}'.format(self.moves[-1]))
            elif not sf.legal_moves(self.variant, self.get_start_fen(), self.moves):
                game_end = True
                result = sf.game_result(self.variant, self.get_start_fen(), self.moves)
            else:
                game_end, result = sf.is_optional_game_end(self.variant, self.get_start_fen(), self.moves)
            if game_end:
                self.result = int(math.copysign(1, result if len(self.moves) % 2 == 0 else -result)) if result else result
            return game_end

    def play(self):
        self.initialize()
        # Both engines start out in force mode and take over their color on
        # their first turn, see Engine.newgame.
        activated = [False, False]
        # The partner board sets its own result, we only have to notice it. Do
        # not evaluate the partner's position from this thread: that used to
        # race with the partner appending its move.
        while not self.is_game_end() and not (self.partner and self.partner.result is not None):
            with self.lock:
                idx = len(self.moves) % 2
                last_move = self.moves[-1] if self.moves else None
                clock, opp_clock = self.clock_times[idx], self.clock_times[idx - 1]
            engine = self.engines[idx]
            engine.update_clocks(clock, opp_clock)
            if last_move:
                engine.usermove(last_move, activate=not activated[idx])
            else:
                engine.go()
            activated[idx] = True
            start_time = time.time()
            # Never wait for an engine forever, otherwise a hung engine freezes
            # the whole match and the time loss below can never be detected.
            kind, move = engine.get_move(timeout=max(clock, 0) + MOVE_TIMEOUT_GRACE)
            end_time = time.time()
            if kind != 'move':
                self.abandon(kind, move, idx)
                return
            # Resolve the move before appending it, so that the partner thread
            # never sees the raw 0000 placeholder and reports it as illegal.
            if move == '0000':
                with self.lock:
                    start_fen, previous_moves = self.get_start_fen(), list(self.moves)
                moves = sf.legal_moves(self.variant, start_fen, previous_moves)
                pass_candidates = [m for m in moves if Engine.move_from_uci(m, False) == '@@@@']
                if len(pass_candidates) == 1:
                    move = pass_candidates[0]
            # Append the move and charge the clock in one step, so that the
            # partner thread never sees one without the other. The start FEN is
            # taken from the same snapshot, since the partner keeps appending to
            # our holdings and the two have to match.
            with self.lock:
                self.moves.append(move)
                self.clock_times[idx] += self.time_control.increment - (end_time - start_time)
                start_fen, current_moves = self.get_start_fen(), list(self.moves)
            logging.debug('Position: {}, Move: {}'.format(sf.get_fen(self.variant, start_fen, current_moves[:-1]), move))
            if self.partner and move in sf.legal_moves(self.variant, start_fen, current_moves[:-1]):
                captured = self.get_captured(start_fen, current_moves)
                if captured:
                    self.partner.set_holdings(captured)

    def abandon(self, kind, claim, idx):
        """Ends the game when the engine to move did not deliver a move."""
        if kind == 'result':
            logging.warning('Engine {} claims the result {}.'.format(idx + 1, claim))
            if claim == '1/2-1/2':
                result = 0
            else:
                # engines[0] moves first, so it has the side to move of the start position
                first_is_white = self.get_start_fen().split()[1] == 'w'
                result = 1 if (claim == '1-0') == first_is_white else -1
            with self.lock:
                self.result = result
            return
        if kind == 'timeout':
            logging.warning('Engine {} did not move in time.'.format(idx + 1))
        elif kind == 'eof':
            logging.error('Engine {} died.'.format(idx + 1))
        elif kind == 'resign':
            logging.info('Engine {} resigns.'.format(idx + 1))
        # The engine to move loses. Game.result is from engines[0]'s point of view.
        with self.lock:
            self.result = -1 if idx == 0 else 1

    def get_captured(self, start_fen, moves):
        previous_fen = sf.get_fen(self.variant, start_fen, moves[:-1])
        current_fen = sf.get_fen(self.variant, start_fen, moves)
        piece_filter = str.isupper if current_fen.split()[1] == 'w' else str.islower
        previous_pieces = self.count_pieces(previous_fen.split()[0], piece_filter)
        current_pieces = self.count_pieces(current_fen.split()[0], piece_filter)
        return ''.join((previous_pieces - current_pieces).elements())

    @staticmethod
    def count_pieces(board, piece_filter):
        """Counts the pieces of one side on the board, demoting promoted ones.

        Promoted pieces are marked with a trailing '~' in the FEN. Capturing one
        of them yields a pawn, so that is what has to be passed to the partner.
        """
        pieces = Counter()
        for piece, suffix in zip(board, board[1:] + ' '):
            if piece_filter(piece):
                pieces[('P' if piece.isupper() else 'p') if suffix == '~' else piece] += 1
        return pieces

    def get_start_fen(self):
        if self.partner:
            # inject obtained holdings
            fen_split = self.start_fen.split(']', 1)
            with self.lock:
                return ']'.join([fen_split[0] + self.obtained_holdings, fen_split[1]])
        else:
            return self.start_fen

    def get_holdings(self):
        fen = sf.get_fen(self.variant, self.get_start_fen(), self.moves)
        return fen.split('[', 1)[1].split(']', 1)[0]

    def set_holdings(self, captured):
        with self.lock:
            self.obtained_holdings += captured
            holdings = self.get_holdings()
        white_holdings = ''.join(filter(str.isupper, holdings)).upper()
        black_holdings = ''.join(filter(str.islower, holdings)).upper()
        xboard_holdings = '[{}] [{}]'.format(white_holdings, black_holdings)
        if captured:
            xboard_holdings += ' ' + ('W' if captured.isupper() else 'B') + captured.upper()
        logging.debug('holding {}'.format(xboard_holdings))
        for engine in self.engines:
            engine.holding(xboard_holdings)


def sprt(score, elo0, elo1):
    """Runs the sequential probability ratio test on a W/L/D score."""
    return stat_util.SPRT({'wins': score[0], 'losses': score[1], 'draws': score[2]},
                          elo0, SPRT_ALPHA, elo1, SPRT_BETA, SPRT_DRAWELO)


def format_elo(score):
    """Formats the Elo estimate, or nothing while it is not defined yet."""
    try:
        elo, elo95, los = stat_util.get_elo(score)
    except (ValueError, ZeroDivisionError):
        # Undefined until the score has enough spread, e.g. before the first loss
        return ''
    if elo95 <= 0:
        # One of the bounds ran past a score of 0 or 1, where Elo is not
        # defined, so there is no interval to report around the estimate yet.
        return ' ELO: {:.2f} (95% interval undefined) LOS: {:.1f}%'.format(elo, 100 * los)
    return ' ELO: {:.2f} +-{:.1f} (95%) LOS: {:.1f}%'.format(elo, elo95, 100 * los)


def format_sprt(score, elo0, elo1):
    result = sprt(score, elo0, elo1)
    return ' LLR: {:.2f} ({:.2f},{:.2f}) [{:.2f},{:.2f}]'.format(
        result['llr'], result['lower_bound'], result['upper_bound'], elo0, elo1)


class Scoreboard:
    """The W/L/D counters and the stop condition shared by all concurrent matches.

    Matches claim the games they are about to play via reserve(), so that
    running several of them concurrently never overshoots the requested number.
    """

    def __init__(self, games, elos=None):
        """Plays up to games games, or until an SPRT on the (elo0, elo1) elos finishes."""
        self.games = games
        self.elos = elos
        self.score = [0, 0, 0]
        self.reserved = 0
        self.stopped = False
        self.lock = threading.Lock()

    def reserve(self, count):
        """Claims up to count games and returns how many were granted."""
        with self.lock:
            if self.stopped or (self.elos and sprt(self.score, *self.elos)['finished']):
                return 0
            granted = max(min(count, self.games - self.reserved), 0)
            self.reserved += granted
            return granted

    def release(self, count):
        """Returns games claimed but not played, e.g. after an aborted match."""
        with self.lock:
            self.reserved -= count

    def record(self, result):
        with self.lock:
            self.score[result - 1] += 1
            score = list(self.score)
        logging.info('Total: {} W: {} L: {} D: {}{}{}'.format(
            sum(score), score[0], score[1], score[2], format_elo(score),
            format_sprt(score, *self.elos) if self.elos else ''))

    def report(self):
        """Logs the final result, and how an SPRT test ended."""
        with self.lock:
            score = list(self.score)
        logging.info('Finished after {} games. W: {} L: {} D: {}{}'.format(
            sum(score), score[0], score[1], score[2], format_elo(score)))
        if self.elos:
            state = sprt(score, *self.elos)['state']
            logging.info('SPRT [{:.2f},{:.2f}]: {}'.format(
                self.elos[0], self.elos[1],
                {'accepted': 'H1 accepted', 'rejected': 'H0 accepted'}.get(state, 'inconclusive')))

    def stop(self):
        """Keeps the matches from starting further games."""
        with self.lock:
            self.stopped = True


class Match:
    def __init__(self, engine1, engine2, e1_options, e2_options, time_control, variant='chess',
                 scoreboard=None, start_fens=None, name='match', max_restarts=MAX_ENGINE_RESTARTS):
        self.two_boards = sf.two_boards(variant)
        self.name = name
        # The names end up on the engines' reader threads, so that the log of a
        # concurrent run says which match and board a line belongs to.
        self.engines = [Engine([engine1], e1_options, '{}:b1e1'.format(name)),
                        Engine([engine2], e2_options, '{}:b1e2'.format(name))]
        self.board2_engines = ([Engine([engine1], e1_options, '{}:b2e1'.format(name)),
                                Engine([engine2], e2_options, '{}:b2e2'.format(name))]
                               if self.two_boards else None)
        self.time_control = time_control
        self.variant = variant
        self.scoreboard = scoreboard if scoreboard is not None else Scoreboard(1)
        self.start_fens = start_fens if start_fens else [sf.start_fen(variant)]
        self.max_restarts = max_restarts
        self.restarts = 0
        for engine in self.all_engines():
            engine.initialize()

    def all_engines(self):
        return self.engines + (self.board2_engines or [])

    def restart_dead_engines(self):
        """Gives every engine that died a fresh process before the next game.

        The Engine objects are reused, so the partner wiring that play_game sets
        up survives a restart. Raises once the restart budget is exhausted, so
        that an engine crashing in every game does not restart forever.
        """
        for engine in self.all_engines():
            if engine.is_alive():
                continue
            if self.restarts >= self.max_restarts:
                raise RuntimeError('Engine {} died and the limit of {} engine restarts is exhausted'
                                   .format(engine.name, self.max_restarts))
            self.restarts += 1
            logging.warning('Engine restart {} of at most {}.'.format(self.restarts, self.max_restarts))
            engine.restart()

    def close(self):
        for engine in self.all_engines():
            engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def play_game(self, flip, start_fen):
        """Plays one game and returns its result from engine 1's point of view.

        Both boards of a two-board variant start from the same position, the
        way a real bughouse game does.
        """
        game = Game(self.engines[flip], self.engines[not flip], self.time_control,
                    self.variant, start_fen)
        if self.two_boards:
            game2 = Game(self.board2_engines[not flip], self.board2_engines[flip], self.time_control,
                         self.variant, start_fen)
            game.partner = game2
            game2.partner = game
            self.engines[0].partner = self.board2_engines[0]
            self.engines[1].partner = self.board2_engines[1]
            self.board2_engines[0].partner = self.engines[0]
            self.board2_engines[1].partner = self.engines[1]
            board2_error = []

            def play_board2():
                try:
                    game2.play()
                except Exception as e:  # noqa: BLE001 - reported in the main thread below
                    logging.exception('Board 2 failed')
                    board2_error.append(e)

            thread2 = threading.Thread(target=play_board2, daemon=True,
                                       name='{}:board2'.format(self.name))
            thread2.start()
        game.play()
        if self.two_boards:
            # Board 2 can only be as late as one search plus the timeout grace.
            # The bound has to come from its clocks rather than from the base
            # time, since an increment lets them grow well beyond it.
            with game2.lock:
                timeout = max(max(game2.clock_times), 0) + 2 * MOVE_TIMEOUT_GRACE
            thread2.join(timeout=timeout)
            if thread2.is_alive():
                raise RuntimeError('Board 2 did not finish within {:.0f}s'.format(timeout))
            if board2_error:
                raise board2_error[0]
            logging.debug('Board1: {}, Board2: {}'.format(game.result, game2.result))
            assert game.result in (1, 0, -1) or game2.result in (1, 0, -1)
            result = (game.result or 0) - (game2.result or 0)
            if result:
                result //= abs(result)
        else:
            result = game.result
        assert result in (1, 0, -1)
        logging.debug('Game finished after {} moves.'.format(len(game.moves)))
        return -result if flip else result

    def run(self):
        """Plays games until the scoreboard is exhausted.

        Games are played in pairs from the same start position with the colors
        swapped, so that neither engine benefits from a lopsided opening.
        """
        while True:
            games = self.scoreboard.reserve(2)
            if not games:
                break
            start_fen = random.choice(self.start_fens)
            for flip in range(games):
                try:
                    self.restart_dead_engines()
                    result = self.play_game(flip, start_fen)
                except BaseException:
                    self.scoreboard.release(games - flip)
                    raise
                self.scoreboard.record(result)


def book_path(book, variant):
    """Resolves --book to a path.

    Given without one, it means the variant's book in a books directory next to
    the script, which is where variantfishtest keeps its collection.
    """
    if book is True:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'books', variant + '.epd')
    return book


def load_book(book, variant):
    """Reads the start positions of an EPD opening book."""
    book = book_path(book, variant)
    two_boards = sf.two_boards(variant)
    fens = []
    with open(book) as epd:
        for number, line in enumerate(epd, start=1):
            fen = line.strip().rstrip(';')
            if not fen:
                continue
            # A two-board variant needs the holdings in the position, since the
            # partner board keeps appending the pieces it passes on to them.
            if sf.validate_fen(fen, variant) != sf.FEN_OK or (two_boards and ']' not in fen):
                logging.warning('Skipping invalid position in {} line {}: {}'.format(book, number, fen))
                continue
            fens.append(fen)
    if not fens:
        raise ValueError('No valid positions in {}'.format(book))
    logging.info('Book {}: {} positions'.format(book, len(fens)))
    return fens


def warn_if_oversubscribed(variant, concurrency):
    """Warns when the concurrent games need more cores than the machine has.

    The clocks are kept in wall clock time and a late reply is counted as a
    loss on time, so oversubscribing the CPUs does not merely add noise to the
    result, it decides games.
    """
    engines_per_game = 4 if sf.two_boards(variant) else 2
    processes = engines_per_game * concurrency
    cpus = os.cpu_count() or 1
    if processes > cpus:
        logging.warning('{} concurrent {} games run {} engines but only {} CPUs are available. Engines that '
                        'do not get a core in time lose on time, which distorts the result. Consider '
                        '--concurrency {}.'
                        .format(concurrency, variant, processes, cpus, max(cpus // engines_per_game, 1)))


def run_match(match, errors):
    """Runs one match in its own thread, ending the whole run if it fails."""
    try:
        match.run()
    except Exception as e:  # noqa: BLE001 - re-raised in the main thread
        logging.exception('Match failed')
        errors.append(e)
        match.scoreboard.stop()


def main(engine1, engine2, e1_options, e2_options, time_control, variant, num_games, concurrency,
         book, sprt, elo0, elo1, **kwargs):
    time_control.warn_if_inexact()
    warn_if_oversubscribed(variant, concurrency)
    start_fens = load_book(book, variant) if book else None
    scoreboard = Scoreboard(num_games, (elo0, elo1) if sprt else None)
    errors = []
    with contextlib.ExitStack() as stack:
        threads = []
        for i in range(concurrency):
            name = 'match{}'.format(i + 1)
            match = stack.enter_context(Match(engine1, engine2, dict(e1_options), dict(e2_options),
                                              time_control, variant, scoreboard, start_fens, name))
            threads.append(threading.Thread(target=run_match, args=(match, errors), name=name, daemon=True))
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logging.warning('Interrupted, waiting for the games in progress to finish.')
            scoreboard.stop()
            for thread in threads:
                thread.join()
    scoreboard.report()
    if errors:
        raise errors[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine1', help='path to first engine')
    parser.add_argument('engine2', help='path to second engine')
    parser.add_argument('--e1-options', help='options for first UCI engine', type=lambda kv: kv.split('='), action='append', default=[])
    parser.add_argument('--e2-options', help='options for second UCI engine', type=lambda kv: kv.split('='), action='append', default=[])
    parser.add_argument('-t', '--time-control', type=str, default='10+0',
                        help='Time control in format moves/time+increment')
    parser.add_argument('-v', '--variant', default='chess', help='variant name')
    parser.add_argument('-n', '--num-games', type=int, default=1000, help='maximum number of games')
    parser.add_argument('-c', '--concurrency', type=int, default=1, help='number of games to play concurrently')
    parser.add_argument('-b', '--book', nargs='?', const=True,
                        help='EPD opening book, or the variant book in books/ if given without a path')
    parser.add_argument('-s', '--sprt', action='store_true', help='stop as soon as an SPRT test concludes')
    parser.add_argument('--elo0', type=float, default=0, help='null hypothesis of the SPRT test')
    parser.add_argument('--elo1', type=float, default=10, help='alternative hypothesis of the SPRT test')
    parser.add_argument('-l', '--log-level', default='INFO', help='logging level')
    parser.add_argument('--strict-xboard', action='store_true',
                        help='announce only whole-second increments, as the xboard protocol defines them')
    args = parser.parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        parser.error('Invalid log level: {}'.format(args.log_level))
    try:
        args.time_control = TimeControl.parse(args.time_control, args.strict_xboard)
    except Exception:
        parser.error('Invalid time control: {}'.format(args.time_control))
    if args.time_control.moves:  # TODO: support epochs
        parser.error('Time control not supported: {}'.format(args.time_control))
    if args.time_control.time < TimeControl.MIN_BASE_TIME:
        parser.error('Base time must be at least {} second, otherwise the engine is left without a '
                     'time limit: {}'.format(TimeControl.MIN_BASE_TIME, args.time_control))
    if args.concurrency < 1:
        parser.error('Invalid concurrency: {}'.format(args.concurrency))
    if args.sprt and args.elo0 >= args.elo1:
        parser.error('elo0 must be below elo1: {} >= {}'.format(args.elo0, args.elo1))
    if args.book and not os.path.exists(book_path(args.book, args.variant)):
        parser.error('Opening book not found: {}'.format(book_path(args.book, args.variant)))
    # Tell the games apart once more than one of them writes to the log
    log_format = '%(threadName)s: %(message)s' if args.concurrency > 1 else '%(message)s'
    logging.basicConfig(level=numeric_level, format=log_format)
    main(**vars(args))
