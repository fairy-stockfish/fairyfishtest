#!/usr/bin/env python3
#
# combine_results sums up the --json results of several fairyfishtest runs,
# e.g. from different machines, into one Elo and SPRT result.
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
import json
import sys

import fairyfishtest
import stat_util


# The results that have to match for the runs to measure the same thing
MATCHING_FIELDS = ('engine1', 'engine2', 'variant', 'time_control', 'book')

# The counters that are summed up, the second value being the length of a list counter
COUNTERS = {'wins': None, 'losses': None, 'draws': None, 'pairs': None, 'engine_restarts': None,
            'pentanomial': 5, 'time_losses': 2, 'crashes': 2, 'illegal_moves': 2}


def load(path):
    with open(path) as result:
        return json.load(result)


def combine(results):
    """Sums the counters of the results. The results are expected to be of the same match."""
    total = {}
    for field, length in COUNTERS.items():
        if length:
            total[field] = [sum(result.get(field, [0] * length)[i] for result in results) for i in range(length)]
        else:
            total[field] = sum(result.get(field, 0) for result in results)
    total['games'] = total['wins'] + total['losses'] + total['draws']
    return total


def mismatches(results):
    """Lists the fields the results disagree on, to catch runs of different matches being mixed up."""
    return ['{}: {}'.format(field, ', '.join(sorted({str(result.get(field)) for result in results})))
            for field in MATCHING_FIELDS if len({result.get(field) for result in results}) > 1]


def report(total, elo0=None, elo1=None):
    """Formats the combined result as lines of text."""
    score = [total['wins'], total['losses'], total['draws']]
    lines = ['Total: {} W: {} L: {} D: {}{}'.format(total['games'], *score, fairyfishtest.format_elo(score))]
    if total['pairs']:
        try:
            elo, elo95, los = stat_util.get_elo_pentanomial(total['pentanomial'])
            lines.append('Pairs: {} {} ELO: {:.2f} +-{:.1f} (95%) LOS: {:.1f}%'.format(
                total['pairs'], total['pentanomial'], elo, elo95, 100 * los))
        except (ValueError, ZeroDivisionError):
            lines.append('Pairs: {} {}'.format(total['pairs'], total['pentanomial']))
    lines.append('Time losses: {}  Crashes: {}  Illegal moves: {}  Engine restarts: {}  (engine 1, engine 2)'.format(
        total['time_losses'], total['crashes'], total['illegal_moves'], total['engine_restarts']))
    if elo0 is not None:
        result = fairyfishtest.sprt(score, elo0, elo1, total['pentanomial'])
        lines.append('SPRT [{:.2f},{:.2f}]:{} {}'.format(
            elo0, elo1, fairyfishtest.format_sprt(score, elo0, elo1, total['pentanomial']),
            {'accepted': 'H1 accepted', 'rejected': 'H0 accepted'}.get(result['state'], 'inconclusive')))
    return lines


def main():
    parser = argparse.ArgumentParser(description='Sum up the --json results of several fairyfishtest runs.')
    parser.add_argument('results', nargs='+', metavar='FILE', help='JSON results of fairyfishtest runs')
    parser.add_argument('--elo0', type=float, help='null hypothesis of the SPRT test, enables it with --elo1')
    parser.add_argument('--elo1', type=float, help='alternative hypothesis of the SPRT test')
    parser.add_argument('-f', '--force', action='store_true',
                        help='sum up the results even if they differ in engines, variant, time control or book')
    parser.add_argument('-o', '--output', metavar='FILE', help='also write the combined result as JSON')
    args = parser.parse_args()
    if (args.elo0 is None) != (args.elo1 is None):
        parser.error('--elo0 and --elo1 have to be given together')
    if args.elo0 is not None and args.elo0 >= args.elo1:
        parser.error('elo0 must be below elo1: {} >= {}'.format(args.elo0, args.elo1))
    results = [load(path) for path in args.results]
    differences = mismatches(results)
    if differences:
        message = 'The results are not from the same match:\n  ' + '\n  '.join(differences)
        if not args.force:
            parser.error(message + '\nPass --force to sum them up anyway.')
        print('Warning: ' + message, file=sys.stderr)
    total = combine(results)
    if not total['games']:
        parser.error('The results contain no games')
    print('\n'.join(report(total, args.elo0, args.elo1)))
    if args.output:
        for field in MATCHING_FIELDS:
            total[field] = results[0].get(field)
        total['version'] = 1
        with open(args.output, 'w') as output:
            json.dump(total, output, indent=2, sort_keys=True)
            output.write('\n')


if __name__ == '__main__':
    main()
