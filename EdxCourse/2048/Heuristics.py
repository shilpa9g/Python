
from Grid import Grid
import PlayerAI

def monotone(state):
    mono = 0

    for x in range(state.size):
        for y in range(state.size-1):
            if state.map[x][y] <= state.map[x][y+1]:
                mono = mono + 1

    for y in range(state.size):
        for x in range(state.size-1):
            if state.map[x][y] >= state.map[x+1][y]:
                mono = mono + 1

    return mono

def free_tiles(state):
    c = getAvailableCells(state)
    free_tiles_h = len(c)
    return free_tiles_h

def max_tile(state):
    maxTile = getMaxTile(state)
    for num in [4,8,16,32,64,128,256,512,1024,2048,4096]:
        if maxTile >= num:
            weight = weight + 2
    return maxTile

def evalheu(state):
    heuristics = (monotone(state)*30/100) + (free_tiles(state)*35/100) + (max_tile(state)*35/100)
    return heuristics