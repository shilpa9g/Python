# Search algorithms

"""
8-puzzle
Python 3
Use a launch.json file to run this file with configuration and arguments

"""

from collections import deque
import timeit
import resource
import sys
import math
import heapq
import itertools

#### SKELETON CODE ####
# The Class that Represents the Puzzle
class PuzzleState:
    """docstring for PuzzleState
    The class definition here is very important. 
    It holds all the information on parent nodes to do the traceback.
    Cost of paths is also evaluated here. 
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []
        # Determine row and column for 0
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break
    # goal-state for an nXn puzzle
    def goal_test(self):
        goal = []
        for i in range(len(self.config)):
            goal.append(i)
        for i in range(len(goal)):
            if goal[i] != self.config[i]:
                return False
        return True

    # Actual moves happening. New configs achieved.
    # Actual actions recorded. Cost increased.
    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            # assigning new values to indexes
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
        return self.children


def bfs_search(hard_state):
    """BFS search with deque and set
    A regular queue is too slow and results in long run-times that can go on for hours.
    Deque is faster.
    Checking for membership in 2 separate sets.(No lists, no single statement checking)
    Checking for membership using only configs is much faster, than the entire puzzle class.
    FIFO
    """
    start = timeit.default_timer()
    frontier = deque()
    frontier.append(hard_state)
    explored = set()
    front = set()
    front.add(hard_state.config)
    max_d = 0

    while len(frontier) > 0:
        state = frontier.popleft()

        if state.goal_test():
            stop = timeit.default_timer()
            trace_back(state, explored, max_d, start, stop)
            return 

        explored.add(state.config)
        for child in state.expand():
            if child.config not in explored:
                if child.config not in front: 
                    frontier.append(child)
                    front.add(child.config)
                    max_d = child.cost

    return

def dfs_search(hard_state):
    """DFS search with deque and set
    Check for membership in reversed order for FILO - dsf

    """
    start = timeit.default_timer()
    frontier = deque()
    frontier.append(hard_state)
    explored = set()
    front = set()
    front.add(hard_state.config)
    max_d = 0

    while len(frontier) > 0:
        state = frontier.pop()
        front.remove(state.config)

        if state.goal_test():
            stop = timeit.default_timer()
            trace_back(state, explored, max_d, start, stop)
            return 

        explored.add(state.config)
        for child in reversed(state.expand()):
            if child.config not in explored:
                if child.config not in front: 
                    frontier.append(child)
                    front.add(child.config)
                    if child.cost > max_d:
                        max_d = child.cost

    return

def trace_back(state, explored, max_d, start, stop):
    path = []
    while state.parent != None: # Back-up till the root. Till no parent.
        path.append(state.action)
        state = state.parent
    path.reverse() # Path from root to goal.
    cost_of_path = len(path)
    nodes_expanded = len(explored)
    search_depth = len(path)
    max_search_depth = max_d
    running_time = format((stop - start), '.8f')
    max_ram_usage = format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024.0, '.8f')
    with open("output.txt", "w") as outF:
        outF.write(" ".join(["path_to_goal:", str(path), "\n"]))
        outF.write(" ".join(["cost_of_path:", str(cost_of_path), "\n"]))
        outF.write(" ".join(["nodes_expanded:", str(nodes_expanded), "\n"]))
        outF.write(" ".join(["search_depth:", str(search_depth), "\n"]))
        outF.write(" ".join(["max_search_depth:", str(max_search_depth), "\n"]))
        outF.write(" ".join(["running_time:", str(running_time), "\n"]))
        outF.write(" ".join(["max_ram_usage:", str(max_ram_usage)]))
    return

  

def A_star_search(hard_state):
    """A * search"""
    start = timeit.default_timer()
    frontier = list() # Use Heap
    heapq.heapify(frontier)
    explored = set()
    front = set()
    #entry_finder = {}
    #REMOVED = '<removed-task>'
    counter = itertools.count() # Second priority check for similar entries
    heuristic = calculate_manhattan_dist(hard_state.n, hard_state.config)
    entry = (heuristic, counter, hard_state) # Tuple for heap
    #entry_finder[hard_state.config] = entry
    heapq.heappush(frontier, entry)
    front.add(hard_state.config)
    max_d = 0

    while len(frontier) > 0:
        entry = heapq.heappop(frontier) # state = frontier.deleteMin()
        h, c, state = entry # Storing individual values from the tuple - to get config
        front.remove(state.config)

        if state.goal_test():
            stop = timeit.default_timer()
            trace_back(state, explored, max_d, start, stop)
            return 

        explored.add(state.config)
        for child in state.expand():
            if child.config not in explored:
                man_dist = calculate_manhattan_dist(child.n, child.config)
                heuristic = child.cost + man_dist 
                count = next(counter)
                entry = (heuristic, count, child) 
                if child.config not in front:
                #if child.config not in entry_finder:
                    heapq.heappush(frontier, entry)
                    front.add(child.config)
                    
                else:
                    for entry in frontier:
                        h, c, state = entry # Storing individual values from the tuple - to check
                        if child.config == state.config:
                            if heuristic < h:
                                entry = (heuristic, count, child) # Replace with lesser heuristic
                                heapq.heapify(frontier) # Heapify again to maintain the heap invariant
                                break
                            break
                if child.cost > max_d:
                        max_d = child.cost

    return

def calculate_manhattan_dist(n, config):
    """calculate the manhattan distance of a state
    abs - absolute value, no negatives
    // and % determine rows and colums to move
    """
    man_dist = 0
    goal = []
    for i in range(len(config)):
        goal.append(i)
    for i, item in enumerate(config):
            if i != item and item != 0:
                man_dist = man_dist + abs(i//n - item//n) + abs(i%n - item%n)
    return man_dist

def main():
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)
    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")
    with open("output.txt" , "r") as f:
        print(f.read())

if __name__ == '__main__':
    main()