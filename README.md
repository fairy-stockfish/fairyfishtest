# fairyfishtest

[![Build Status](https://github.com/ianfab/fairyfishtest/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/ianfab/fairyfishtest/actions/workflows/test.yml?query=branch%3Amaster)

Fairyfishtest is a script for automated chess variant engine matches based on the [python wrapper](https://github.com/gbtami/Fairy-Stockfish) for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish).

It supports the same set of variants as Fairy-Stockfish, including user-defined variants as well as four-player variants like bughouse.

## Features

- **Fixed game count testing**: Traditional testing with a predetermined number of games
- **SPRT (Sequential Probability Ratio Test)**: Statistical testing that can terminate early when sufficient evidence is gathered

## Setup
Run `pip install -r requirements.txt` to install dependencies.

## Usage
Run `python fairyfishtest.py -h` to get info on required and optional arguments.

### Basic Usage (Fixed Games)
```
python fairyfishtest.py stockfish-test stockfish-base -v bughouse -n 100 -t 10+0
```

### SPRT Usage
For SPRT testing, provide all five SPRT parameters:
```
python fairyfishtest.py stockfish-test stockfish-base -v chess -n 1000 -t 10+0 \
  --sprt-elo0 0 --sprt-elo1 5 --sprt-alpha 0.05 --sprt-beta 0.05 --sprt-drawelo 200
```

**SPRT Parameters:**
- `--sprt-elo0`: Null hypothesis ELO (H0) - typically 0 for "no improvement" 
- `--sprt-elo1`: Alternative hypothesis ELO (H1) - target improvement (e.g., 5 ELO)
- `--sprt-alpha`: Maximum Type I error rate (false positive) - typically 0.05
- `--sprt-beta`: Maximum Type II error rate (false negative) - typically 0.05  
- `--sprt-drawelo`: Draw ELO parameter for probability calculations - typically 200

When using SPRT, the test will terminate early if there's sufficient statistical evidence to accept or reject the null hypothesis, potentially saving significant testing time compared to fixed game counts.
