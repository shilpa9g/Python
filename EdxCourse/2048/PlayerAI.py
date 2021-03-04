# Code submitted for grading - Max Tile 512
from BaseAI import BaseAI
import ComputerAI
from Grid import Grid
import operator
import time
import secrets
import copy

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)

class PlayerAI(BaseAI, Grid): #multiple inheritance
    # Define required variables ex: self.action, self.children
    
    def __init__(self, board=[[0] * 4 for i in range(4)],
    action=0, utility=0, depth = 0):

        self.utility = utility
        self.action = action
        self.depth = depth
        self.children = []
        self.size = 4
        #self.map = [[0] * self.size for i in range(self.size)]
        self.map = board #[0] * self.size for i in range(self.size)]
        self.size = 4

    def moove(self, action):

        if action == 0:
            return self.moveUpDown(False)
        if action == 1:
            return self.moveUpDown(True)
        if action == 2:
            return self.moveLeftRight(False)
        if action == 3:
            return self.moveLeftRight(True)

    # Move Up or Down
    def moveUpDown(self, down):
    
        board = self.map
        r = range(self.size -1, -1, -1) if down else range(self.size)
        #r = range(self.size)

        pass

        for j in range(self.size):
            cells = []

            for i in r:
                cell = board.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.Merge(cells)

            for i in r:
                value = cells.pop(0) if cells else 0
                board.map[i][j] = value
        return self


    # move left or right
    def moveLeftRight(self, right):
        board = self.map
        r = range(self.size - 1, -1, -1) if right else range(self.size)

        pass

        for i in range(self.size):
            cells = []

            for j in r:
                cell = board.map[i][j]

                if cell != 0:
                    cells.append(cell)

            self.Merge(cells)

            for j in r:
                value = cells.pop(0) if cells else 0
                board.map[i][j] = value
        return self

    # Merge Tiles
    def Merge(self, cells):
        if len(cells) <= 1:
            return cells

        i = 0

        while i < len(cells) - 1:
            if cells[i] == cells[i+1]:
                cells[i] *= 2

                del cells[i+1]

            i += 1


    def Clone(self):
        PlayerCopy = PlayerAI()
        PlayerCopy.map = Grid()
        PlayerCopy.map = copy.deepcopy(self.map)
        PlayerCopy.size = self.size

        return PlayerCopy

    def max_children(self):    
        board = self.map
        moves = board.getAvailableMoves() 
        for action in moves:
            PlayerCopy = self.Clone()
            new = PlayerCopy.moove(action) # to-do : check for the actual child grid config
            utility = evalheu(new.map, action)
            child = PlayerAI(new.map, action, utility, self.depth+1)
            self.children.append(child)
        #sorted(self.children, key=itemgetter(2), reverse=True)
        self.children.sort(key=operator.attrgetter('utility'))
        #self.children = reversed(self.children)
        return self.children

    def min_children(self):
        board = self.map
        cells = board.getAvailableCells()
        for i in range(3):
            pos = secrets.choice(cells)
            PlayerCopy = self.Clone()
            board = PlayerCopy.map
            board.map[pos[0]][pos[1]] = 2

            utility = evalheu(board, 4)
            entry = PlayerAI(board, 2, utility, self.depth+1)
            self.children.append(entry)
            #board = self.map
        self.children.sort(key=operator.attrgetter('utility'))
        return self.children

    def getMove(self, grid):
        state = PlayerAI() # converting to another class object
        state.map = grid
        start_time = time.perf_counter()
        (result, utility) = maximise(state, -1, 150000, start_time)
        return result.action


def maximise(state, alpha, beta, start_time): # player
    empty = PlayerAI()   
    if time.perf_counter() - start_time > 0.2:
        return (state, state.utility)

    maxChild, maxUtility = state, alpha

    for child in state.max_children():
        (state, utility) = minimise(child, alpha, beta, start_time)
                
        if utility > maxUtility:
            (maxChild, maxUtility) = (state, utility)
                    
        if maxUtility >= beta:
            break
                    
        if maxUtility > alpha:
            alpha = maxUtility
                    
    return (maxChild, maxUtility)


def minimise(state, alpha, beta, start_time): # computer
    empty = PlayerAI()
    if time.perf_counter() - start_time > 0.2:
        return (state, state.utility)

    minChild, minUtility = state, 150000

    for child in state.min_children():
        (state, utility) = maximise(child, alpha, beta, start_time)
                
        if utility < minUtility:
            (minChild, minUtility) = (state, utility)
                    
        if minUtility <= alpha:
            break
                    
        if minUtility < beta:
            beta = minUtility
                    
    return (minChild, minUtility)


def monotone(state):
    mono = 0
    for x in range(4):
        for y in range(3):
            if state.map[x][y] <= state.map[x][y+1] and state.map[x][y] != 0: 
                mono = mono + 5
    if state.getMaxTile() > 60:
        mono = mono * 1.5

    for y in range(4):
        for x in range(3):
            if state.map[x][y] >= state.map[x+1][y] and state.map[x][y] != 0:
                mono = mono + 15

    return mono

def free_tiles(state):
    #board = state.map
    c = state.getAvailableCells()
    if len(c) > 6: # from 8 to 6
        free_tiles_h = len(c)*20
    else:
        free_tiles_h = len(c)*80
    return free_tiles_h

def max_tile(state):
    #board = state.map
    maxTile = state.getMaxTile()
    weight = 1 # changed 0 to 1
    for num in [32,64,128,256,512,1024,2048,4096]:
        if maxTile >= num:
            weight = weight * 2 # changed from 10 to 2
    return maxTile

def adjacentTile(state, action):

    value = 0
    for x in range(state.size):
        for y in range(state.size):

            # If Current Cell is Filled
            if state.map[x][y] and action < 4:

                # Look Ajacent Cell Value
                move = directionVectors[action]

                adjCellValue = state.getCellValue((x + move[0], y + move[1]))

                # If Value is the Same or Adjacent Cell is Empty
                if adjCellValue == state.map[x][y]:
                    value = value + 10
                    if state.map[x][y] >= 32: #64 to 60
                        value = value * 4 
                        if state.map[x][y] >= 128: #-------
                            value = value * 4 #----------
    if len(state.getAvailableCells()) < 8:
        value = value * 2
    return value


def evalheu(board, action):
    '''
    heuristics = (monotone(board)*10) + (free_tiles(board)*3.5) + (max_tile(board))
    + (adjacentTile(board, action))
    '''
    heuristics = (monotone(board)*3) + (free_tiles(board)*3.5) + (max_tile(board)*5)
    + (adjacentTile(board, action))
    return heuristics