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
import logging
import math
import random
import subprocess
import sys
import threading

import pyffish as sf


class Engine:
    def __init__(self, args):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        self.lock = threading.Lock()

    def initialize(self):
        with self.lock:
            self.process.stdin.write('xboard\n')
            self.process.stdin.write('protover 2\n')
            self.process.stdin.flush()

    def newgame(self, variant):
        with self.lock:
            self.process.stdin.write('new\n')
            self.process.stdin.write('variant {}\n'.format(variant))
            # TODO: implement time management
            self.process.stdin.write('sd 10\n')
            self.process.stdin.flush()

    def go(self):
        with self.lock:
            self.process.stdin.write('go\n')
            self.process.stdin.flush()

    def get_move(self):
        while True:
            l = self.process.stdout.readline()
            # TODO: communicate ptell
            if l.startswith('move'):
                return l.strip().split()[1]
            if l.startswith(('1-0', '0-1', '1/2-1/2')):
                return None

    def holding(self, holding):
        with self.lock:
            self.process.stdin.write('holding {}\n'.format(holding))
            self.process.stdin.flush()

    def usermove(self, move):
        with self.lock:
            self.process.stdin.write('usermove {}\n'.format(move))
            self.process.stdin.flush()


class Game:
    def __init__(self, engine1, engine2, variant='chess', start_fen=None):
        self.engines = [engine1, engine2]
        self.variant = variant
        self.start_fen = start_fen or sf.start_fen(variant)
        self.moves = []
        self.result = None
        self.partner = None
        self.obtained_holdings = ''
        self.lock = threading.RLock()

    def initialize(self):
        for engine in self.engines:
            engine.initialize()
            engine.newgame(self.variant)

    def is_game_end(self):
        with self.lock:
            game_end = False
            if self.moves and self.moves[-1] not in sf.legal_moves(self.variant, self.get_start_fen(), self.moves[:-1]):
                # last move was illegal
                result = 1
                game_end = True
                logging.error('Illegal move: {}'.format(self.moves[-1]))
            elif not sf.legal_moves(self.variant, self.get_start_fen(), self.moves):
                game_end, result = sf.is_immediate_game_end(self.variant, self.get_start_fen(), self.moves)
                if not game_end:
                    # TODO: handle conditional checkmate, and non-standard checkmate and stalemate values
                    if sf.gives_check(self.variant, self.get_start_fen(), self.moves):
                        result = -1
                    else:
                        result = 0
                    game_end = True
            else:
                game_end, result = sf.is_optional_game_end(self.variant, self.get_start_fen(), self.moves)
            if game_end:
                self.result = int(math.copysign(1, result if len(self.moves) % 2 == 0 else -result)) if result else result
            return game_end

    def play(self):
        self.initialize()
        while not self.is_game_end() and not (self.partner and self.partner.is_game_end()):
            engine = self.engines[len(self.moves) % 2]
            engine.usermove(self.moves[-1]) if self.moves else engine.go()
            with self.lock:
                self.moves.append(engine.get_move())
            logging.debug('Position: {}, Move: {}'.format(sf.get_fen(self.variant, self.get_start_fen(), self.moves),
                                                          self.moves[-1]))
            if self.partner:
                captured = self.get_captured()
                if captured:
                    self.partner.set_holdings(captured)

    def get_captured(self):
        # TODO: this does not consider promoted pieces
        previous_fen = sf.get_fen(self.variant, self.get_start_fen(), self.moves[:-1])
        current_fen = sf.get_fen(self.variant, self.get_start_fen(), self.moves)
        piece_filter = str.isupper if current_fen.split()[1] == 'w' else str.islower
        previous_pieces = Counter(filter(piece_filter, previous_fen.split()[0]))
        current_pieces = Counter(filter(piece_filter, current_fen.split()[0]))
        return ''.join((previous_pieces - current_pieces).elements())

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

class Match:
    def __init__(self, engine1, engine2, variant='chess', games=1, start_fens=None):
        self.two_boards = variant in ('bughouse', 'koedem', 'supply') # TODO: use sf.twoBoards
        self.engines = [Engine([engine1]), Engine([engine2])]
        self.board2_engines = [Engine([engine1]), Engine([engine2])] if self.two_boards else None
        self.variant = variant
        self.games = games
        self.start_fens = start_fens if start_fens else [sf.start_fen(variant)]
        self.score = [0, 0, 0]

    def play_game(self):
        flip = sum(self.score) % 2
        game = Game(self.engines[flip], self.engines[not flip],
                    self.variant, random.choice(self.start_fens))
        if self.two_boards:
            game2 = Game(self.board2_engines[not flip], self.board2_engines[flip],
                         self.variant, random.choice(self.start_fens))
            game.partner = game2
            game2.partner = game
            thread2 = threading.Thread(target=game2.play, daemon=True)
            thread2.start()
        game.play()
        if self.two_boards:
            thread2.join()
            logging.debug('Board1: {}, Board2: {}'.format(game.result, game2.result))
            assert game.result in (1, 0, -1) or game2.result in (1, 0, -1)
            result = (game.result or 0) - (game2.result or 0)
            if result:
                result //= abs(result)
        else:
            result = game.result
        assert result in (1, 0, -1)
        relative_result = -result if flip else result
        self.score[relative_result - 1] += 1
        logging.debug('Game finished after {} moves.'.format(len(game.moves)))

    def run(self):
        while sum(self.score) < self.games:
            self.play_game()
            logging.info('Total: {} W: {} L: {} D: {}'.format(sum(self.score), *self.score))


def main(engine1, engine2, variant, num_games, **kwargs):
    match = Match(engine1, engine2, variant, num_games)
    match.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine1', help='path to first engine', type=str)
    parser.add_argument('engine2', help='path to second engine', type=str)
    parser.add_argument('-v', '--variant', help='variant name', type=str, default='chess')
    parser.add_argument('-n', '--num-games', help='maximum number of games', type=int, default=3)
    parser.add_argument('-l', '--log-level', help='logging level', default='INFO')
    args = parser.parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        parser.error('Invalid log level: {}'.format(args.log_level))
    logging.basicConfig(level=numeric_level, format='%(message)s')
    main(**vars(args))
