student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import math

def f_1(x):
    return ((pow(x, 2) * math.cos(x/10) - x) / (100))

def f_2(x):
    return (math.log(pow(math.sin(x/20), 1/2)))

def f_3(x):
    return (math.log(math.cos(x)+(45/x)))

def draw(func, x_range):
    results = list()
    xs = list()
    for x in x_range:
        results.append(func(x))
        xs.append(x)
    plt.plot(xs, results)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    pass

def gradiant_descent(func, initial_point: float, learning_rate: float, max_iterations: int):
    h = 0.0001
    x = initial_point
    for _ in range(max_iterations):
        x = x - learning_rate * (func(x+h) - func(x)) / h
    return f_1(x)

def f(x_1, x_2):
    return (2 * pow(x_1, 2) + 3 * pow(x_2, 2) - 4 * x_1 * x_2 - 50 * x_1 + 6 * x_2)

def gradiant_descent(func, initial_point: Tuple, learning_rate: float, threshold: float, max_iterations: int):
    x_1_sequence = [initial_point[0]]
    x_2_sequence = [initial_point[1]]
    
    x_1 = initial_point[0]
    x_2 = initial_point[1]
    for _ in range(max_iterations):
        x_1, x_2 = update_points(func, x_1, x_2, learning_rate)
        if x_1 > threshold or x_2 > threshold:
            break
        x_1_sequence.append(x_1)
        x_2_sequence.append(x_2)
    return x_1_sequence, x_2_sequence

def update_points(func, x_1, x_2, learning_rate):
    h = 0.000001
    x_1 = x_1 - (learning_rate * (func(x_1 + h, x_2) - f(x_1, x_2)) / h)
    x_2 = x_2 - (learning_rate * (func(x_1, x_2 + h) - f(x_1, x_2)) / h)
    return x_1, x_2

    
x_1_seqs, x_2_seqs = gradiant_descent(f, initial_point, learning_rates[0], threshold, max_iterations)
draw_points_sequence(f, x_1_seqs, x_2_seqs)


x_1_seqs, x_2_seqs = gradiant_descent(f, initial_point, learning_rates[1], threshold, max_iterations)
draw_points_sequence(f, x_1_seqs, x_2_seqs)

    
x_1_seqs, x_2_seqs = gradiant_descent(f, initial_point, learning_rates[2], threshold, max_iterations)
draw_points_sequence(f, x_1_seqs, x_2_seqs)

    
x_1_seqs, x_2_seqs = gradiant_descent(f, initial_point, learning_rates[3], threshold, max_iterations)
draw_points_sequence(f, x_1_seqs, x_2_seqs)


def initialize_nodes(n, m, values):
    variables = []
    for _ in range(n):
        variables.append([])
    i = 1
    for halls_list in values:
        for hall in halls_list:
            variables[hall - 1].append(i)
        i += 1
    return variables
def initialize_edges(e, edges_input):
    edges = []
    for edge in edges_input:
        edges.append(edge)
    return edges

def initialize_variable_neighbors(n, edges_input):
    variable_neighbors = []
    for _ in range(n):
        variable_neighbors.append([])
    for edge in edges_input:
        variable_neighbors[edge[0] - 1].append(edge[1])
        variable_neighbors[edge[1] - 1].append(edge[0])
    return variable_neighbors

def ac_3(variables, edges):
    
    edges_queue = edges.copy()
    while len(edges_queue) > 0:
        edge = edges_queue.pop(0)
        node_one = edge[0] - 1
        node_two = edge[1] - 1
        node_one_values = variables[node_one]
        node_two_values = variables[node_two]

        removed_inconsistancy = False
        for one_value in node_one_values:
            consistant = False
            for two_value in node_two_values:
                if two_value != one_value:
                    consistant = True
                    break
            if not consistant:
                node_one_values.remove(one_value)
                removed_inconsistancy = True
        
        if removed_inconsistancy:
            for edge in edges:
                if edge[1] == node_one:
                    edges_queue.append(edge)

        

def backtrack(variables, variable_neighbors, assignment, var_index):
    
    if not (0 in assignment):
        return assignment
    if len(variables) <= 0:
        return "NO"
    variable = variables.pop()
    for value in variable:
        neighbors = variable_neighbors[var_index]
        consistant = True
        for neighbor in neighbors:
            if (assignment[neighbor - 1] == value):
                consistant = False
                break
        if consistant:
            assignment[var_index] = value
            result = backtrack(variables, variable_neighbors, assignment, var_index - 1)
            if result != "NO":
                return result
            assignment[var_index] = 0
    return "NO"

    

def backtracking_search(variables, edges, variable_neighbors):
    
    ac_3(variables, edges)
    assignment = []
    for _ in variables:
        assignment.append(0)
    return backtrack(variables, variable_neighbors, assignment, len(variables) - 1)




class MinimaxPlayer(Player):
    
    def __init__(self, col, x, y, depth):
        super().__init__(col, x, y)
        self.depth = depth

    def checkCol(self, x, board):
        
        if x > board.getSize() - 1 or x < 0:
            return False
        return True

    def checkRow(self, y, board):
        
        if y > board.getSize() - 1 or y < 0:
            return False
        return True

    def moveU(self, x, y, board):
        
        return self.checkRow(y+1, board)
    

    def moveD(self, x, y, board):
        
        return self.checkRow(y-1, board)

    def moveR(self, x, y, board):
        
        return self.checkCol(x+1, board)
    
    def moveL(self, x, y, board):
        
        return self.checkCol(x-1, board)

    def moveUR(self, x, y, board):
        
        return self.checkCol(x+1, board) and self.checkRow(y+1, board)
    
    def moveUL(self, x, y, board):
        
        return self.checkCol(x-1, board) and self.checkRow(y+1, board)

    def moveDR(self, x, y, board):
        
        return self.checkCol(x+1, board) and self.checkRow(y-1, board)

    def moveDL(self, x, y, board):
        
        return self.checkCol(x-1, board) and self.checkRow(y-1, board)

    def canMove(self, x, y, board):
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 2:
                return False
            if self.moveU(x, y, board) or \
             self.moveD(x, y, board) or \
             self.moveR(x, y, board) or \
             self.moveL(x, y, board) or \
             self.moveUR(x, y, board) or \
             self.moveUL(x, y, board) or \
             self.moveDR(x, y, board) or \
             self.moveDL(x, y, board):
                return True
    def generate_child(self, board, player_col, x, y):
        child_board = Board(board)
        if child_board.move(IntPair(x, y), player_col) == 0:
            return True, child_board
        return False, None
    
    def generate_children(self, board, player_col):
        x = board.getPlayerX(player_col)
        y = board.getPlayerY(player_col)
        result = []
        childU = self.generate_child(board, player_col, x, y+1)
        if childU[0]:
            result.append(childU[1])
        childUR = self.generate_child(board, player_col, x+1, y+1)
        if childUR[0]:
            result.append(childUR[1])
        childR = self.generate_child(board, player_col, x+1, y)
        if childR[0]:
            result.append(childR[1])
        childRD = self.generate_child(board, player_col, x+1, y-1)
        if childRD[0]:
            result.append(childRD[1])
        childD = self.generate_child(board, player_col, x, y-1)
        if childD[0]:
            result.append(childD[1])
        childDL = self.generate_child(board, player_col, x-1, y-1)
        if childDL[0]:
            result.append(childDL[1])
        childL = self.generate_child(board, player_col, x-1, y)
        if childL[0]:
            result.append(childL[1])
        childLU = self.generate_child(board, player_col, x-1, y+1)
        if childLU[0]:
            result.append(childLU[1])
        return result
        
    def minValue(self, board, alpha, beta, depth):
        
        v = float('inf')
        new_board = board
        children = self.generate_children(board, 2)
        if depth == 0 or len(children) == 0:
            return board.getScore(1), new_board
        for child in children:
            eval, child_board = self.maxValue(child, alpha, beta, depth - 1)

            if eval < v:
                new_board = child

            v = min(v, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return v, new_board    
    
    def maxValue(self, board, alpha, beta, depth):
        
        v = float('-inf')
        new_board = board
        children = self.generate_children(board, 1)
        if depth == 0 or len(children) == 0:
            return board.getScore(1), new_board
        for child in children:
            eval, child_board = self.minValue(child, alpha, beta, depth - 1)

            if eval > v:
                new_board = child

            v = max(v, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return v, new_board
    
    def getMove(self, board):
        
        alpha = float('-inf')
        beta = float('inf')
        next = IntPair(-20, -20)

        if (board.getNumberOfMoves() == board.maxNumberOfMoves):
            return IntPair(-20, -20)
        
        if not (self.canMove(board.getPlayerX(self.getCol()), board.getPlayerY(self.getCol()), board)):
            return IntPair(-10, -10)
        
        if (self.getCol() == 1):
            
            result = self.maxValue(board, alpha, beta, self.depth)
            x_next = result[1].getPlayerX(1)
            y_next = result[1].getPlayerY(1)
            
            return IntPair(x_next, y_next)

        else:
            
            result = self.maxValue(board, alpha, beta, self.depth)
            x_next = result[1].getPlayerX(2)
            y_next = result[1].getPlayerY(2)
            return IntPair(x_next, y_next)

p1 = MinimaxPlayer(1, 0, 0, 4)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)

density = []
for i in range(4):
    p1 = MinimaxPlayer(1, 0, 0, 4)
    p2 = NaivePlayer(2, 7, 7)
    g = Game(p1, p2)
    numberOfMatches = 4
    score1, score2 = g.start(numberOfMatches)
    density.append(score1/numberOfMatches)

plt.plot(density)
plt.xlabel("# round")
plt.ylabel("density win player 1")
plt.show()




density = []
for i in range(3):
    p1 = MinimaxPlayer(1, 0, 0, 3)
    p2 = MinimaxPlayer(2, 7, 7, 3)
    g = Game(p1, p2)
    numberOfMatches = 3
    score1, score2 = g.start(numberOfMatches)
    density.append(score1/numberOfMatches)

plt.plot(density)
plt.xlabel("# round")
plt.ylabel("density win player 1")
plt.show()




density = []
for i in range(5):
    p1 = MinimaxPlayer(1, 0, 0, i)
    p2 = NaivePlayer(2, 7, 7)
    g = Game(p1, p2)
    numberOfMatches = 4
    score1, score2 = g.start(numberOfMatches)
    density.append(score1)

plt.plot(density)
plt.xlabel("depth")
plt.ylabel("#win for player 1")
plt.show()



density = []
for i in range(4):
    p1 = MinimaxPlayer(1, 0, 0, i)
    p2 = MinimaxPlayer(2, 7, 7, 3 - i)
    g = Game(p1, p2)
    numberOfMatches = 3
    score1, score2 = g.start(numberOfMatches)
    density.append(score1)

plt.plot(density)
plt.xlabel("depth")
plt.ylabel("#win for player 1")
plt.show()



