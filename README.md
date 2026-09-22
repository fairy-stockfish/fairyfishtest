# fairyfishtest

[![Build Status](https://github.com/ianfab/fairyfishtest/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/ianfab/fairyfishtest/actions/workflows/test.yml?query=branch%3Amaster)

Fairyfishtest is a script for automated chess variant engine matches based on the [python wrapper](https://github.com/gbtami/Fairy-Stockfish) for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish).

It supports the same set of variants as Fairy-Stockfish, including user-defined variants as well as four-player variants like bughouse.

## Setup
Run `pip install -r requirements.txt` to install dependencies.

## Usage
Run `python fairyfishtest.py -h` to get info on required and optional arguments.

Example:
```
python fairyfishtest.py stockfish-test stockfish-base -v bughouse -n 100 -t 10+0
```

### Concurrency
By default one game is played at a time. `-c/--concurrency` plays several games in parallel, each with its own set of engine processes:
```
python fairyfishtest.py stockfish-test stockfish-base -v bughouse -n 100 -t 10+0 -c 3
```
Note that a two-board variant like bughouse already runs four engines per game, so the number of engine processes is `4 * concurrency`. Since the clocks are kept in wall clock time and a late reply is counted as a loss on time, running more engines than the machine has cores does not just add noise to the result, it decides games. The script warns when the requested concurrency oversubscribes the CPUs.

### Opening books
`-b/--book` reads start positions from an EPD file, one position per line, and picks one at random for each pair of games. Positions that the variant does not accept are skipped with a warning. Given without a path, `-b` looks for `books/<variant>.epd` next to the script, which is where [variantfishtest](https://github.com/ianfab/variantfishtest) keeps its collection.

```
python fairyfishtest.py stockfish-test stockfish-base -v bughouse -n 100 -t 10+0 -b books/bughouse.epd
```

Both boards of a two-board variant start from the same position, the way a real bughouse game does. A book for such a variant has to include the holdings in its positions, since the partner board keeps appending the pieces it passes on to them.

### SPRT and Elo
Every result line carries the Elo estimate, its 95% error bar and the [likelihood of superiority](https://www.chessprogramming.org/Match_Statistics#Likelihood_of_superiority) of the first engine:

```
Total: 100 W: 63 L: 34 D: 3 ELO: 103.73 +-71.1 (95%) LOS: 99.9%
```

`-s/--sprt` stops the run as soon as a [sequential probability ratio test](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test) can decide between the two hypotheses, rather than playing all `-n` games. `--elo0` is the null hypothesis and `--elo1` the alternative, defaulting to 0 and 10:

```
python fairyfishtest.py stockfish-test stockfish-base -v bughouse -n 5000 -t 10+0 -b books/bughouse.epd -s --elo0 0 --elo1 5
```

The result lines then also carry the log-likelihood ratio and the bounds it is tested against, and the run reports which hypothesis it settled on:

```
Total: 317 W: 152 L: 70 D: 95 ELO: 91.96 +-32.7 (95%) LOS: 100.0% LLR: 2.96 (-2.94,2.94) [0.00,10.00]
Finished after 317 games. W: 152 L: 70 D: 95 ELO: 91.96 +-32.7 (95%) LOS: 100.0%
SPRT [0.00,10.00]: H1 accepted
```

### Time controls
The clocks are kept and enforced by the script itself, and announced to the engines in centiseconds through the xboard `time` and `otim` commands before every move. The base time therefore reaches the engines exactly, whatever the `level` command rounded it to.

The increment has no such channel. The xboard protocol only defines it in whole seconds, so a fractional increment such as `-t 10+0.1` is announced as a real number, which needs an engine that reads it as one. Fairy-Stockfish does; other engines read an increment of `0` instead and budget their time accordingly, which is what they did for a fractional increment before as well. Pass `--strict-xboard` to announce the whole-second increment the protocol defines to every engine, at the cost of the engines budgeting for a time control that is not the one being played.

The base time has to be at least one second. Below that the `level` command announces `0:0`, which leaves Fairy-Stockfish without a time limit, and it never returns a move.
