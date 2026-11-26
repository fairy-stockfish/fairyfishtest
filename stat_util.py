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


def get_elo_pentanomial(pentanomial):
    """Returns the Elo difference, its 95% error bar and the likelihood of superiority.

    The counts are those of game pairs with swapped colors by their points
    (0, 0.5, 1, 1.5 and 2). Both games of a pair start from the same position,
    so the pair is the independent sample, which gives a tighter and fairer
    error bar than treating all games as independent when the openings differ
    in how much they favor either color.
    """
    n = sum(pentanomial)
    scores = [points / 4 for points in range(5)]
    probabilities = [float(count) / n for count in pentanomial]
    mu = sum(p * score for p, score in zip(probabilities, scores))
    stdev = math.sqrt(sum(p * (score - mu) ** 2 for p, score in zip(probabilities, scores))) / math.sqrt(n)
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


# Replaces zero counts, so that the empirical distribution is regularized as in fishtest
REGULARIZATION = 1e-3

# Normalized Elo per unit of normalized t-value, 800 / ln(10)
NELO_DIVIDED_BY_NT = 800 / math.log(10)


def _sprt_state(result):
    """Fills in whether the test has finished and how, given its llr and bounds."""
    if result['llr'] < result['lower_bound']:
        result['finished'] = True
        result['state'] = 'rejected'
    elif result['llr'] > result['upper_bound']:
        result['finished'] = True
        result['state'] = 'accepted'
    return result


def _sprt_result(alpha, beta):
    return {
        'finished': False,
        'state': '',
        'llr': 0.0,
        'lower_bound': math.log(beta / (1 - alpha)),
        'upper_bound': math.log((1 - beta) / alpha),
    }


def SPRT(R, elo0, alpha, elo1, beta, drawelo=None):
    """Sequential probability ratio test of H0: elo = elo0 against H1: elo = elo1, on single games.

    alpha is the maximum type I error, reached at elo = elo0, and beta the
    maximum type II error for elo >= elo1, reached at elo = elo1. R holds the
    number of 'wins', 'losses' and 'draws'. The drawelo is estimated out of the
    sample, so the drawelo argument is only kept for compatibility.

    Without any draws the test is on wins and losses alone, with a drawelo of 0.
    A missing win or loss count is regularized, so that the test also runs on
    a sample of only wins or only losses.

    Returns whether the test has 'finished', its 'state' ('accepted' for H1,
    'rejected' for H0, or empty while it runs), the log-likelihood ratio 'llr'
    and the 'lower_bound' and 'upper_bound' it is tested against.
    """
    result = _sprt_result(alpha, beta)
    wins, losses, draws = float(R['wins']), float(R['losses']), float(R['draws'])
    if wins + losses + draws == 0:
        return result
    if wins + losses == 0:
        # Only draws, which tell nothing about the strength
        return result
    wins, losses = max(wins, REGULARIZATION), max(losses, REGULARIZATION)
    n = wins + losses + draws
    probabilities = {'win': wins / n, 'loss': losses / n, 'draw': draws / n}
    if draws > 0:
        _, drawelo = proba_to_bayeselo(probabilities)
    else:
        drawelo = 0.0

    # Probability laws under H0 and H1
    p0 = bayeselo_to_proba(elo0, drawelo)
    p1 = bayeselo_to_proba(elo1, drawelo)

    result['llr'] = (wins * math.log(p1['win'] / p0['win'])
                     + losses * math.log(p1['loss'] / p0['loss']))
    if draws > 0:
        result['llr'] += draws * math.log(p1['draw'] / p0['draw'])

    return _sprt_state(result)


def _secular(pdf):
    """Solves the secular equation sum_i p_i * a_i / (1 + x * a_i) = 0 for x."""
    values = [a for a, _ in pdf]
    v, w = min(values), max(values)
    if v * w >= 0:
        raise ValueError('the support of the distribution does not straddle the expectation')
    epsilon = 1e-9
    lower, upper = -1 / w + epsilon, -1 / v - epsilon

    def f(x):
        return sum(p * a / (1 + x * a) for a, p in pdf)

    # f is strictly decreasing between its poles, so the root can be bisected
    for _ in range(200):
        middle = (lower + upper) / 2
        if f(middle) > 0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def _mle_expected(pdf, s):
    """Maximum likelihood estimate of a distribution on the values of pdf with expectation s."""
    x = _secular([(a - s, p) for a, p in pdf])
    return [(a, p / (1 + x * (a - s))) for a, p in pdf]


def _results_to_pdf(results):
    """Turns counts into the empirical distribution of the score of a game (pair) in [0, 1]."""
    counts = [max(float(count), REGULARIZATION) for count in results]
    n = sum(counts)
    return n, [(i / (len(counts) - 1), count / n) for i, count in enumerate(counts)]


def _logistic(elo_):
    return 1 / (1 + 10 ** (-elo_ / 400))


def LLR_pentanomial(pentanomial, elo0, elo1, elo_model='logistic'):
    """Generalized log-likelihood ratio of H1 against H0 for game pair counts.

    The counts are those of game pairs with swapped colors by their points
    (0, 0.5, 1, 1.5 and 2). Instead of assuming a model for the draws, the
    likelihood of the observed distribution of the pair score is maximized
    under both hypotheses (the GSPRT of Michel Van den Bergh, as in fishtest).

    elo_model 'logistic' takes elo0 and elo1 as logistic Elo, i.e. as
    hypotheses on the expected score. 'normalized' takes them as normalized
    Elo, i.e. on the expected score in units of its standard deviation, with
    the closed-form approximation of the generalized likelihood ratio.
    """
    if len(pentanomial) != 5:
        raise ValueError('expected 5 pentanomial counts')
    n, pdf = _results_to_pdf(pentanomial)
    if elo_model == 'logistic':
        pdf0, pdf1 = (_mle_expected(pdf, _logistic(elo)) for elo in (elo0, elo1))
        return n * sum(p * (math.log(q1) - math.log(q0)) for (_, p), (_, q0), (_, q1) in zip(pdf, pdf0, pdf1))
    if elo_model == 'normalized':
        mu = sum(a * p for a, p in pdf)
        variance = sum(p * (a - mu) ** 2 for a, p in pdf)
        sigma_per_game = math.sqrt(2 * variance)
        nt = (mu - 0.5) / sigma_per_game
        nt0, nt1 = (elo / NELO_DIVIDED_BY_NT for elo in (elo0, elo1))
        return n * math.log((1 + (nt - nt0) ** 2) / (1 + (nt - nt1) ** 2))
    raise ValueError('unknown elo model: {}'.format(elo_model))


def SPRT_pentanomial(pentanomial, elo0, alpha, elo1, beta, elo_model='logistic'):
    """Sequential probability ratio test like SPRT, but on game pairs with swapped colors.

    Both games of a pair start from the same position, so the pair is the
    independent sample. This copes with draws of any frequency, including none.
    pentanomial holds the counts of pairs by their points (0, 0.5, 1, 1.5, 2).
    """
    result = _sprt_result(alpha, beta)
    if sum(pentanomial) == 0:
        return result
    result['llr'] = LLR_pentanomial(pentanomial, elo0, elo1, elo_model)
    return _sprt_state(result)
