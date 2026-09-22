#!/usr/bin/env python3
#
# Statistics for engine matches: Elo estimates and the sequential probability
# ratio test. Ported from variantfishtest, which took it from fishtest.
#
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

import math


def erf_inv(x):
    """Inverse of the error function, from an analytically inverted approximation."""
    a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
    y = math.log(1 - x * x)
    z = 2 / (math.pi * a) + y / 2
    return math.copysign(math.sqrt(math.sqrt(z * z - y / a) - z), x)


def phi(q):
    """Cumulative distribution function for the standard Gaussian law: quantile -> probability."""
    return 0.5 * (1 + math.erf(q / math.sqrt(2)))


def phi_inv(p):
    """Quantile function for the standard Gaussian law: probability -> quantile."""
    assert 0 <= p <= 1
    return math.sqrt(2) * erf_inv(2 * p - 1)


def elo(x):
    if x <= 0:
        return 0.0
    return -400 * math.log10(1 / x - 1)


def get_elo(wld):
    """Returns the Elo difference, its 95% error bar and the likelihood of superiority."""
    n = sum(wld)
    w, l, d = (float(count) / n for count in wld)

    # mu is the empirical mean of the variables (Xi), assumed i.i.d.
    mu = w + d / 2

    # stdev is the empirical standard deviation of the random variable (X1+...+X_N)/N
    stdev = math.sqrt(w * (1 - mu) ** 2 + l * (0 - mu) ** 2 + d * (0.5 - mu) ** 2) / math.sqrt(n)

    # 95% confidence interval for mu
    mu_min = mu + phi_inv(0.025) * stdev
    mu_max = mu + phi_inv(0.975) * stdev

    return elo(mu), (elo(mu_max) - elo(mu_min)) / 2, phi((mu - 0.5) / stdev)


def bayeselo_to_proba(elo_, drawelo):
    """Turns an Elo expressed in BayesElo, relative to drawelo, into win/loss/draw probabilities."""
    probabilities = {
        'win': 1.0 / (1.0 + pow(10.0, (-elo_ + drawelo) / 400.0)),
        'loss': 1.0 / (1.0 + pow(10.0, (elo_ + drawelo) / 400.0)),
    }
    probabilities['draw'] = 1.0 - probabilities['win'] - probabilities['loss']
    return probabilities


def proba_to_bayeselo(probabilities):
    """Turns win/loss probabilities into an Elo expressed in BayesElo, and the drawelo it is relative to."""
    assert 0 < probabilities['win'] < 1 and 0 < probabilities['loss'] < 1
    elo_ = 200 * math.log10(probabilities['win'] / probabilities['loss']
                            * (1 - probabilities['loss']) / (1 - probabilities['win']))
    drawelo = 200 * math.log10((1 - probabilities['loss']) / probabilities['loss']
                               * (1 - probabilities['win']) / probabilities['win'])
    return elo_, drawelo


def SPRT(R, elo0, alpha, elo1, beta, drawelo):
    """Sequential probability ratio test of H0: elo = elo0 against H1: elo = elo1.

    alpha is the maximum type I error, reached at elo = elo0, and beta the
    maximum type II error for elo >= elo1, reached at elo = elo1. R holds the
    number of 'wins', 'losses' and 'draws'.

    Returns whether the test has 'finished', its 'state' ('accepted' for H1,
    'rejected' for H0, or empty while it runs), the log-likelihood ratio 'llr'
    and the 'lower_bound' and 'upper_bound' it is tested against.
    """
    result = {
        'finished': False,
        'state': '',
        'llr': 0.0,
        'lower_bound': math.log(beta / (1 - alpha)),
        'upper_bound': math.log((1 - beta) / alpha),
    }

    # Estimate drawelo out of sample
    if not (R['wins'] > 0 and R['losses'] > 0 and R['draws'] > 0):
        return result
    n = R['wins'] + R['losses'] + R['draws']
    probabilities = {'win': float(R['wins']) / n, 'loss': float(R['losses']) / n, 'draw': float(R['draws']) / n}
    _, drawelo = proba_to_bayeselo(probabilities)

    # Probability laws under H0 and H1
    p0 = bayeselo_to_proba(elo0, drawelo)
    p1 = bayeselo_to_proba(elo1, drawelo)

    result['llr'] = (R['wins'] * math.log(p1['win'] / p0['win'])
                     + R['losses'] * math.log(p1['loss'] / p0['loss'])
                     + R['draws'] * math.log(p1['draw'] / p0['draw']))

    if result['llr'] < result['lower_bound']:
        result['finished'] = True
        result['state'] = 'rejected'
    elif result['llr'] > result['upper_bound']:
        result['finished'] = True
        result['state'] = 'accepted'

    return result
