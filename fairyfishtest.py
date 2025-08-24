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
import threading
import time

import pyffish as sf


# SPRT (Sequential Probability Ratio Test) statistical functions
def erf(x):
    """Error function approximation for SPRT calculations"""
    a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
    x2 = x * x
    y = -x2 * (4 / math.pi + a * x2) / (1 + a * x2)
    return math.copysign(math.sqrt(1 - math.exp(y)), x)


def erf_inv(x):
    """Inverse error function for SPRT calculations"""
    a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
    y = math.log(1 - x * x)
    z = 2 / (math.pi * a) + y / 2
    return math.copysign(math.sqrt(math.sqrt(z * z - y / a) - z), x)


def phi(q):
    """Cumulative distribution function for the standard Gaussian law"""
    return 0.5 * (1 + erf(q / math.sqrt(2)))


def phi_inv(p):
    """Quantile function for the standard Gaussian law"""
    assert (0 <= p <= 1)
    return math.sqrt(2) * erf_inv(2 * p - 1)


def elo(x):
    """Convert win rate to ELO"""
    if x <= 0:
        return 0.0
    return -400 * math.log10(1 / x - 1)


def bayeselo_to_proba(elo, drawelo):
    """Convert BayesELO to probabilities"""
    P = {}
    P['win'] = 1.0 / (1.0 + pow(10.0, (-elo + drawelo) / 400.0))
    P['loss'] = 1.0 / (1.0 + pow(10.0, (elo + drawelo) / 400.0))
    P['draw'] = 1.0 - P['win'] - P['loss']
    return P


def proba_to_bayeselo(P):
    """Convert probabilities to BayesELO"""
    assert (0 < P['win'] < 1 and 0 < P['loss'] < 1)
    elo = 200 * math.log10(P['win'] / P['loss'] * (1 - P['loss']) / (1 - P['win']))
    drawelo = 200 * math.log10((1 - P['loss']) / P['loss'] * (1 - P['win']) / P['win'])
    return elo, drawelo


def SPRT(R, elo0, alpha, elo1, beta, drawelo):
    """Sequential Probability Ratio Test
    
    Args:
        R: dict with 'wins', 'losses', 'draws' counts
        elo0: null hypothesis ELO
        alpha: max type I error
        elo1: alternative hypothesis ELO  
        beta: max type II error
        drawelo: draw ELO
        
    Returns:
        dict with 'finished', 'state', 'llr', 'lower_bound', 'upper_bound'
    """
    result = {
        'finished': False,
        'state': '',
        'llr': 0.0,
        'lower_bound': math.log(beta / (1 - alpha)),
        'upper_bound': math.log((1 - beta) / alpha),
    }

    # Estimate drawelo out of sample
    if R['wins'] > 0 and R['losses'] > 0 and R['draws'] > 0:
        N = R['wins'] + R['losses'] + R['draws']
        P = {'win': float(R['wins']) / N, 'loss': float(R['losses']) / N, 'draw': float(R['draws']) / N}
        elo, drawelo = proba_to_bayeselo(P)
    else:
        return result

    # Probability laws under H0 and H1
    P0 = bayeselo_to_proba(elo0, drawelo)
    P1 = bayeselo_to_proba(elo1, drawelo)

    # Log-Likelihood Ratio
    result['llr'] = (R['wins'] * math.log(P1['win'] / P0['win']) + 
                     R['losses'] * math.log(P1['loss'] / P0['loss']) + 
                     R['draws'] * math.log(P1['draw'] / P0['draw']))

    if result['llr'] < result['lower_bound']:
        result['finished'] = True
        result['state'] = 'rejected'
    elif result['llr'] > result['upper_bound']:
        result['finished'] = True
        result['state'] = 'accepted'

    return result


class Engine:
    def __init__(self, args, options):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        self.lock = threading.Lock()
        self.partner = None
        self.rank_conversion = False  # convert ranks from zero-based to one-based
        self.options = options

    def initialize(self):
        with self.lock:
            self.process.stdin.write('xboard\n')
            self.process.stdin.write('protover 2\n')
            for option, value in self.options.items():
                self.process.stdin.write('option {}={}\n'.format(option, value))
            self.process.stdin.flush()

    def newgame(self, variant, time_control):
        with self.lock:
            self.rank_conversion = sf.start_fen(variant).count('/') + 1 == 10
            self.process.stdin.write('new\n')
            self.process.stdin.write('variant {}\n'.format(variant))
            if self.partner:
                self.process.stdin.write('partner test\n')
            self.process.stdin.write('level {}\n'.format(time_control.format_xboard()))
            self.process.stdin.flush()

    def update_clocks(self, time, otim):
        with self.lock:
            # times in centiseconds
            self.process.stdin.write('time {}\n'.format(int(time * 100)))
            self.process.stdin.write('otim {}\n'.format(int(otim * 100)))
            self.process.stdin.flush()

    def go(self):
        with self.lock:
            self.process.stdin.write('go\n')
            self.process.stdin.flush()

    def get_move(self):
        while True:
            line = self.process.stdout.readline()
            if self.partner and line.startswith('tellics ptell'):
                self.partner.ptell(line.strip().split(None, 2)[2])
            if line.startswith('move'):
                return self.move_to_uci(line.strip().split()[1], self.rank_conversion)
            if line.startswith(('1-0', '0-1', '1/2-1/2')):
                return None

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
        with self.lock:
            self.process.stdin.write('ptell {}\n'.format(message))
            self.process.stdin.flush()

    def holding(self, holding):
        with self.lock:
            self.process.stdin.write('holding {}\n'.format(holding))
            self.process.stdin.flush()

    def usermove(self, move):
        with self.lock:
            self.process.stdin.write('usermove {}\n'.format(self.move_from_uci(move, self.rank_conversion)))
            self.process.stdin.flush()


class TimeControl:
    def __init__(self, time, increment=0, moves=0):
        self.time = time
        self.increment = increment
        self.moves = moves

    @staticmethod
    def parse(stringified_tc):
        time_and_increment = stringified_tc.split('+')
        moves_and_time = time_and_increment[0].split('/')
        time = float(moves_and_time[-1])
        increment = float(time_and_increment[1]) if len(time_and_increment) > 1 else 0
        moves = int(moves_and_time[0]) if len(moves_and_time) > 1 else 0
        return TimeControl(time, increment, moves)

    def format_xboard(self):
        return '{} {}:{} {}'.format(self.moves, int(self.time // 60), int(self.time % 60), int(self.increment))

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
        for engine in self.engines:
            engine.initialize()
            engine.newgame(self.variant, self.time_control)

    def is_legal(self):
        assert self.moves
        return self.moves[-1] in sf.legal_moves(self.variant, self.get_start_fen(), self.moves[:-1])

    def is_game_end(self):
        with self.lock:
            game_end = False
            if self.clock_times[(len(self.moves) - 1) % 2] <= 0:
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
        while not self.is_game_end() and not (self.partner and self.partner.is_game_end()):
            idx = len(self.moves) % 2
            engine = self.engines[idx]
            engine.update_clocks(self.clock_times[idx], self.clock_times[idx - 1])
            engine.usermove(self.moves[-1]) if self.moves else engine.go()
            start_time = time.time()
            with self.lock:
                self.moves.append(engine.get_move())
            # woraround for e1e1-style passing moves
            if self.moves[-1] == '0000':
                moves = sf.legal_moves(self.variant, self.get_start_fen(), self.moves[:-1])
                pass_candidates = [move for move in moves if Engine.move_from_uci(move, False) == '@@@@']
                if len(pass_candidates) == 1:
                    self.moves[-1] = pass_candidates[0]
            end_time = time.time()
            self.clock_times[idx] += self.time_control.increment - (end_time - start_time)
            logging.debug('Position: {}, Move: {}'.format(sf.get_fen(self.variant, self.get_start_fen(), self.moves[:-1]),
                                                          self.moves[-1]))
            if self.partner and self.is_legal():
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
    def __init__(self, engine1, engine2, e1_options, e2_options, time_control, variant='chess', games=1, start_fens=None, sprt_params=None):
        self.two_boards = sf.two_boards(variant)
        self.engines = [Engine([engine1], e1_options), Engine([engine2], e2_options)]
        self.board2_engines = [Engine([engine1], e1_options), Engine([engine2], e2_options)] if self.two_boards else None
        self.time_control = time_control
        self.variant = variant
        self.games = games
        self.start_fens = start_fens if start_fens else [sf.start_fen(variant)]
        self.score = [0, 0, 0]  # [losses, draws, wins] for engine1 vs engine2
        self.sprt_params = sprt_params
        self.sprt_result = None

    def play_game(self):
        flip = sum(self.score) % 2
        game = Game(self.engines[flip], self.engines[not flip], self.time_control,
                    self.variant, random.choice(self.start_fens))
        if self.two_boards:
            game2 = Game(self.board2_engines[not flip], self.board2_engines[flip], self.time_control,
                         self.variant, random.choice(self.start_fens))
            game.partner = game2
            game2.partner = game
            self.engines[0].partner = self.board2_engines[0]
            self.engines[1].partner = self.board2_engines[1]
            self.board2_engines[0].partner = self.engines[0]
            self.board2_engines[1].partner = self.engines[1]
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
            
            # Check SPRT termination conditions
            if self.sprt_params and sum(self.score) >= 3:  # Need at least 3 games for SPRT
                R = {
                    'wins': self.score[2],      # wins for engine1
                    'losses': self.score[0],    # losses for engine1
                    'draws': self.score[1]      # draws
                }
                self.sprt_result = SPRT(R, **self.sprt_params)
                
                if self.sprt_result['finished']:
                    logging.info('SPRT finished: {} (LLR: {:.3f}, bounds: [{:.3f}, {:.3f}])'.format(
                        self.sprt_result['state'], 
                        self.sprt_result['llr'],
                        self.sprt_result['lower_bound'],
                        self.sprt_result['upper_bound']
                    ))
                    break
                else:
                    logging.debug('SPRT continuing: LLR: {:.3f}, bounds: [{:.3f}, {:.3f}]'.format(
                        self.sprt_result['llr'],
                        self.sprt_result['lower_bound'],
                        self.sprt_result['upper_bound']
                    ))


def main(engine1, engine2, e1_options, e2_options, time_control, variant, num_games, sprt_elo0=None, sprt_elo1=None, sprt_alpha=None, sprt_beta=None, sprt_drawelo=None, **kwargs):
    sprt_params = None
    if all(x is not None for x in [sprt_elo0, sprt_elo1, sprt_alpha, sprt_beta, sprt_drawelo]):
        sprt_params = {
            'elo0': sprt_elo0,
            'elo1': sprt_elo1,
            'alpha': sprt_alpha,
            'beta': sprt_beta,
            'drawelo': sprt_drawelo
        }
        logging.info('Using SPRT with parameters: H0={}, H1={}, alpha={}, beta={}, drawelo={}'.format(
            sprt_elo0, sprt_elo1, sprt_alpha, sprt_beta, sprt_drawelo))
    
    match = Match(engine1, engine2, dict(e1_options), dict(e2_options), time_control, variant, num_games, sprt_params=sprt_params)
    match.run()


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
    parser.add_argument('-l', '--log-level', default='INFO', help='logging level')
    
    # SPRT arguments
    parser.add_argument('--sprt-elo0', type=float, help='SPRT null hypothesis ELO (H0)')
    parser.add_argument('--sprt-elo1', type=float, help='SPRT alternative hypothesis ELO (H1)')
    parser.add_argument('--sprt-alpha', type=float, help='SPRT maximum Type I error rate')
    parser.add_argument('--sprt-beta', type=float, help='SPRT maximum Type II error rate')
    parser.add_argument('--sprt-drawelo', type=float, help='SPRT draw ELO')
    
    args = parser.parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        parser.error('Invalid log level: {}'.format(args.log_level))
    try:
        args.time_control = TimeControl.parse(args.time_control)
    except Exception:
        parser.error('Invalid time control: {}'.format(args.time_control))
    if args.time_control.moves:  # TODO: support epochs
        parser.error('Time control not supported: {}'.format(args.time_control))
    
    # Validate SPRT parameters
    sprt_args = [args.sprt_elo0, args.sprt_elo1, args.sprt_alpha, args.sprt_beta, args.sprt_drawelo]
    sprt_provided = sum(1 for x in sprt_args if x is not None)
    if sprt_provided > 0 and sprt_provided != 5:
        parser.error('All SPRT parameters must be provided together: --sprt-elo0, --sprt-elo1, --sprt-alpha, --sprt-beta, --sprt-drawelo')
    
    logging.basicConfig(level=numeric_level, format='%(message)s')
    main(**vars(args))
