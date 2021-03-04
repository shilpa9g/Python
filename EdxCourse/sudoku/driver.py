# Sudoku 

import math
import sys
from collections import deque
import copy

class AC3_Function:
    def __init__(self, board, arcs, main_board, main_arcs):
        self.copy_sudoku = board
        self.copy_pairs = arcs
        self.sudoku = main_board
        self.cell_pairs = main_arcs

    def AC3(self):
        output = []
        while self.copy_pairs:
            (Xi, Xj) = self.copy_pairs.pop() 
            if self.Revise(Xi, Xj):
                if self.sudoku[Xi] == {}:
                    return False
                for Xk in self.XiNeighbors(Xi, Xj):
                    pair = (Xk, Xi)
                    self.copy_pairs.append(pair)
                self.copy_sudoku[Xi] = self.sudoku[Xi]
        for key, value in self.copy_sudoku.items():
            if len(value) == 1:
                output.append(value[0])
            else:
                #print(self.copy_sudoku)
                #return "AC3"
                bts = self.BTS()
                for key, value in self.copy_sudoku.items():
                    output.append(self.copy_sudoku[key][0])
                #output = self.copy_sudoku.values()
                algo = " BTS"
                print(self.copy_sudoku)
                print("bts")
                return output, algo
        algo = " AC3"
        print(self.copy_sudoku)
        print("ac3")
        return output, algo

    def Revise(self, Xi, Xj):
        revised = False
        for x in self.copy_sudoku[Xi]:
            c = 0
            for y in self.copy_sudoku[Xj]:
                if x != y:
                    c += 1
                    break
            if c == 0:
                self.sudoku[Xi].remove(x)
                revised = True
        #copy_sudoku[Xi] = sudoku[Xi]
        return revised

    def XiNeighbors(self, Xi, Xj): # remove Xj
        neighbors = []
        set_cells = set(self.cell_pairs)
        for entry in set_cells:
            if Xi == entry[1] and entry[0] != Xj:
                neighbors.append(entry[0])
        return neighbors


    def BTS(self):
        key, value =  self.if_full()
        if value == 0:
            return True
        else:
            #for key, value in self.copy_sudoku.items():
                #if len(value) != 1:
            for try_value in value:
                check = list()
                check.append(try_value)
                if self.conditions(key, check):
                    self.copy_sudoku[key] = check

                    if self.BTS():
                        return True
                                
                self.copy_sudoku[key] = value
                #return

        return False

    def if_full(self):
        
        for key, value in self.copy_sudoku.items():
            if len(value) != 1:
                
                sorted_sudoku = sorted(self.copy_sudoku.items(), key=lambda kv: (len(kv[1]), kv[0]))
                while sorted_sudoku:
                    if len(sorted_sudoku[0][1]) < 2:
                        sorted_sudoku.remove(sorted_sudoku[0])
                    else:
                        break
                while sorted_sudoku:
                    return sorted_sudoku[0][0], sorted_sudoku[0][1]
                
                #return key, value
        return 'Done', 0              

    def conditions(self, key, check):
        neighbors = self.XiNeighbors(key, key)
        for neighbor in neighbors:
            if self.copy_sudoku[neighbor] != check:
                condition = True
            else:
                condition = False
                break

        return condition

        




def main():        
    values = sys.argv[1]

    values = tuple(map(int, tuple(values)))
    #print(values)
    n = int(math.sqrt(len(values)))
    domain = tuple(range(1,n+1))
    #print(n, domain)
    sudoku = {}
    count = 0
    letters = tuple("ABCDEFGHI")
    for letter in letters:
        for num in domain:
            cell = letter + str(num)
            if values[count] != 0:
                number = list()
                number.append(values[count])
                sudoku[cell] = number
            else:
                sudoku[cell] = list(domain)
            count += 1
    #print(sudoku)

    cell_pairs = deque()


    # queue of row arcs
    for letter in letters:
        for i, num in enumerate(domain):
            for j, num1 in enumerate(domain):
                if j != n-1 and i != n-1 and j>=i:
                    cell1 = letter + str(num)
                    cell2 = letter + str(domain[j+1])
                    pair1 = cell1, cell2
                    pair2 = cell2, cell1
                    cell_pairs.append(pair1)
                    cell_pairs.append(pair2)

    # queue of column arcs
    for num in domain:
        for i, letter in enumerate(letters):
            for j, letter1 in enumerate(letters):
                if letter != 'I' and letter1 != 'I' and j>=i:
                    cell1 = letter + str(num)
                    cell2 = letters[j+1] + str(num)
                    pair1 = cell1, cell2
                    pair2 = cell2, cell1
                    cell_pairs.append(pair1)
                    cell_pairs.append(pair2)
    #print(cell_pairs)


    # queue of 3X3 arcs
    cells = []
    pairs = []
    s, e = 0, 3
    sn, en = 0, 3

    while e <= 9:
        while en <= 9:
            for letter in letters[s:e]:
                for num in domain[sn:en]:
                    cell = letter + str(num)
                    cells.append(cell)
            #print(cells)
            for cell in cells:
                for cell1 in cells:
                    set_cells = set(cell_pairs)
                    if cell != cell1 and (cell, cell1) not in set_cells:
                        if (cell1, cell) not in set_cells:
                            pair1 = cell, cell1
                            pair2 = cell1, cell
                            cell_pairs.append(pair1)
                            cell_pairs.append(pair2)
            sn += 3
            en += 3
            cells = []
        s += 3
        e += 3
        sn, en = 0, 3
        cells = []
    #print(cell_pairs)

    copy_sudoku = copy.deepcopy(sudoku)
    copy_pairs = copy.deepcopy(cell_pairs)
    solver = AC3_Function(copy_sudoku, copy_pairs, sudoku, cell_pairs)
    result, al = solver.AC3()
    string = ''.join([str(elem) for elem in result]) 
    print(string)
    with open("output.txt", "w") as outF:
        outF.write("".join([string, al]))
    #print (result)
    #if result:
        #print(solver.copy_sudoku)

if __name__ =='__main__':
    main()


