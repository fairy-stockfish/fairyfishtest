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

### Time controls
The clocks are kept and enforced by the script itself, and announced to the engines in centiseconds through the xboard `time` and `otim` commands before every move. The base time therefore reaches the engines exactly, whatever the `level` command rounded it to.

The increment has no such channel. The xboard protocol only defines it in whole seconds, so a fractional increment such as `-t 10+0.1` is announced as a real number, which needs an engine that reads it as one. Fairy-Stockfish does; other engines read an increment of `0` instead and budget their time accordingly, which is what they did for a fractional increment before as well. Pass `--strict-xboard` to announce the whole-second increment the protocol defines to every engine, at the cost of the engines budgeting for a time control that is not the one being played.

The base time has to be at least one second. Below that the `level` command announces `0:0`, which leaves Fairy-Stockfish without a time limit, and it never returns a move.
