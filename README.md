![logo](https://github.com/dlee888/detractor/blob/main/assets/detractor.svg)

# DeTractor

This codebase provides an environment for the card game of Tractor, along with code for RL experiments learning to play.

Simplifications:
- Rounds are ignored. The trump rank is always 2, and the objective is simply to maximize the number of points won within one round.
- The first player to get dealt a 2 is assumed to always declare trump.
- Kitty exchange is automatically done via a heuristic to simplify the agent's task.
