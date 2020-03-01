# fairyfishtest
Fairyfishtest is a script for automated chess variant engine matches based on the [python wrapper](https://github.com/gbtami/Fairy-Stockfish) for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish).

It supports the same set of variants as Fairy-Stockfish, including user-defined variants as well as four-player variants like bughouse.

## Setup
Run `pip install -r requirements.txt` to install dependencies.

## Usage
Run `python fairyfishtest.py -h` to get info on required and optional arguments.

## Limitations
The project is still in an experimental stage and has several limitations:
* Setting time controls is not supported yet, only a fixed depth limit can be used for testing.
* Only CECP protocol is supported, not UCI/USI/UCCI.
* Some rules are not fully supported yet, like non-standard checkmate and stalemate values or demotion of captured promoted pawns in bughouse.