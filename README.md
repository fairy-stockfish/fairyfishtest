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
