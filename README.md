# LuMaPython

Battlesnake AI Agent using a hybrid approach, combining dynamic heuristics with logit quantal response equilibrium. The agent is written in Python and was developed as part of a university class at LUH. 

## Abilities

- Layered heuristics: Finds best field to aim for considering general favorability of fields, food, distance and enemy positions
- Path planning: Efficient path planning using modified A*, considering potential dangers by avoiding unfavorable fields
- Close combat mode: Uses quantal response equilibrium to approximate Nash equilibrium and make decisions considering future enemy decisions
- Can deal with limited information (i.e. in special game mode where only area with radius of 3 fields around head is visible), estimating the current game state (including own strength and enemy strenghts)

## Success

Placed first in internal competition for the university course. Peak position on Battlesnakes standard mode leaderboard: 13. Currently, the agent is no longer active.
