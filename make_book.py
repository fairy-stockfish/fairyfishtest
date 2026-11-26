#!/usr/bin/env python3
#
# make_book generates an opening book for fairyfishtest for any variant.
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
import concurrent.futures
import logging
import queue
import random
import subprocess
import threading

import pyffish as sf

from fairyfishtest import get_captured


class UciEngine:
    """The little of the UCI protocol that is needed to ask an engine for an evaluation."""

    def __init__(self, path, variant, options=()):
        self.process = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        universal_newlines=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._read_loop, daemon=True).start()
        self.send('uci')
        self.wait_for('uciok')
        self.send('setoption name UCI_Variant value {}'.format(variant))
        for name, value in options:
            self.send('setoption name {} value {}'.format(name, value))
        self.send('isready')
        self.wait_for('readyok')

    def _read_loop(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)  # EOF

    def send(self, line):
        self.process.stdin.write(line + '\n')
        self.process.stdin.flush()

    def wait_for(self, prefix, timeout=60):
        """Reads the engine's output up to the line starting with prefix and returns the lines read."""
        read = []
        while True:
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError('Engine did not answer within {}s'.format(timeout))
            if line is None:
                raise RuntimeError('Engine died')
            read.append(line)
            if line.startswith(prefix):
                return read

    def evaluate(self, fen, go_command):
        """Returns the score of the position in centipawns for the side to move, None for a mate."""
        self.send('position fen {}'.format(fen))
        self.send(go_command)
        score = None
        for line in self.wait_for('bestmove', timeout=600):
            fields = line.split()
            if fields[:1] == ['info'] and 'score' in fields and 'lowerbound' not in fields \
                    and 'upperbound' not in fields:
                kind, value = fields[fields.index('score') + 1:fields.index('score') + 3]
                score = int(value) if kind == 'cp' else None
        return score

    def close(self):
        try:
            self.send('quit')
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass


def random_opening(variant, plies, rng):
    """Plays random legal moves from the start position and returns the position, or None.

    None is returned for a line that ended early. For a two-board variant the
    holdings of the position are those the other board would have passed on in
    a game like this: every capture is credited to the side that made it, since
    a position without any holdings is not what a game of such a variant looks
    like, and fairyfishtest adds the pieces the partner board passes on to them.
    """
    start_fen = sf.start_fen(variant)
    two_boards = sf.two_boards(variant)
    moves, holdings = [], ''
    for _ in range(plies):
        legal = sf.legal_moves(variant, start_fen, moves)
        if not legal:
            return None
        moves.append(rng.choice(legal))
        if two_boards:
            holdings += get_captured(variant, start_fen, moves).swapcase()
    if not sf.legal_moves(variant, start_fen, moves) or sf.is_optional_game_end(variant, start_fen, moves)[0]:
        return None
    fen = sf.get_fen(variant, start_fen, moves)
    if two_boards:
        head, tail = fen.split(']', 1)
        fen = head + holdings + ']' + tail
        if sf.validate_fen(fen, variant) != sf.FEN_OK or not sf.legal_moves(variant, fen, []):
            return None
    return fen


def generate(engine_factory, variant, count, min_plies, max_plies, max_score, go_command, rng,
             workers=1, max_attempts=None):
    """Collects count distinct random openings that the engine evaluates within max_score centipawns."""
    max_attempts = max_attempts or 200 * count
    book, seen = [], set()
    lock = threading.Lock()
    attempts = [0]
    local = threading.local()
    engines = []

    def work(seed):
        thread_rng = random.Random(seed)
        if not hasattr(local, 'engine'):
            local.engine = engine_factory()
            with lock:
                engines.append(local.engine)
        while True:
            with lock:
                if len(book) >= count or attempts[0] >= max_attempts:
                    return
                attempts[0] += 1
            fen = random_opening(variant, thread_rng.randint(min_plies, max_plies), thread_rng)
            if fen is None:
                continue
            with lock:
                if fen in seen:
                    continue
                seen.add(fen)
            score = local.engine.evaluate(fen, go_command)
            if score is None or abs(score) > max_score:
                continue
            with lock:
                if len(book) < count:
                    book.append(fen)
                    logging.info('{}/{} {} {:+d}'.format(len(book), count, fen, score))

    try:
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for future in [pool.submit(work, rng.random()) for _ in range(workers)]:
                future.result()
    finally:
        for engine in engines:
            engine.close()
    return book, attempts[0]


def main():
    parser = argparse.ArgumentParser(
        description='Generate an opening book for fairyfishtest: random openings of a few plies, '
                    'kept if the engine evaluates them as balanced.')
    parser.add_argument('engine', help='path to a UCI engine that supports the variant')
    parser.add_argument('-v', '--variant', default='chess', help='variant name')
    parser.add_argument('-n', '--num-positions', type=int, default=1000, help='number of positions to generate')
    parser.add_argument('-o', '--output', required=True, metavar='FILE', help='EPD file to write')
    parser.add_argument('--min-plies', type=int, default=4, help='minimum number of random plies')
    parser.add_argument('--max-plies', type=int, default=8, help='maximum number of random plies')
    parser.add_argument('--max-score', type=int, default=100,
                        help='keep positions evaluated within this many centipawns of equality')
    parser.add_argument('--depth', type=int, default=10, help='search depth of the evaluation')
    parser.add_argument('--movetime', type=int, help='search time of the evaluation in milliseconds, instead of depth')
    parser.add_argument('--options', type=lambda kv: kv.split('=', 1), action='append', default=[],
                        metavar='NAME=VALUE', help='UCI option of the engine, can be given several times')
    parser.add_argument('-j', '--workers', type=int, default=1, help='number of engine processes to evaluate with')
    parser.add_argument('--seed', type=int, help='seed of the random number generator, for reproducible books')
    parser.add_argument('--max-attempts', type=int, help='give up after this many random openings')
    parser.add_argument('-l', '--log-level', default='INFO', help='logging level')
    args = parser.parse_args()
    if args.variant not in sf.variants():
        parser.error('Unknown variant: {}'.format(args.variant))
    if not 0 <= args.min_plies <= args.max_plies:
        parser.error('Invalid ply range: {}-{}'.format(args.min_plies, args.max_plies))
    if args.num_positions < 1 or args.workers < 1:
        parser.error('The number of positions and workers must be positive')
    logging.basicConfig(level=args.log_level.upper(), format='%(message)s')
    go_command = 'go movetime {}'.format(args.movetime) if args.movetime else 'go depth {}'.format(args.depth)
    rng = random.Random(args.seed)
    book, attempts = generate(lambda: UciEngine(args.engine, args.variant, args.options), args.variant,
                              args.num_positions, args.min_plies, args.max_plies, args.max_score, go_command,
                              rng, args.workers, args.max_attempts)
    if len(book) < args.num_positions:
        logging.warning('Only {} of {} positions found after {} attempts.'.format(len(book), args.num_positions,
                                                                                  attempts))
    if not book:
        parser.error('No balanced positions found, try a larger --max-score or fewer plies')
    with open(args.output, 'w') as output:
        output.write(''.join(fen + '\n' for fen in book))
    logging.info('Wrote {} positions to {}.'.format(len(book), args.output))


if __name__ == '__main__':
    main()
