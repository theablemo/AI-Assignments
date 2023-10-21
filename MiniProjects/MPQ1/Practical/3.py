import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt
"""
No need to change this cell. You can change "./Inputs/test-q3-q4.txt" to test different graphs.
"""

graph_matrix =[]
def load_data(path = "./Inputs/test-q3-q4.txt"):
    with  open(path , 'r') as f:
        lines = f.readlines()
        number_of_vertices = int(lines[0])
        for i in range(number_of_vertices):
            line_split = lines[i+1].split(',');
            graph_matrix.append([])
            for j in range(number_of_vertices):
                graph_matrix[i].append(int(line_split[j]))
load_data()
edge_count = int(np.sum(graph_matrix)/2)
print(edge_count)
def random_state_generator(n):
    res = []
    for _ in range(n):
        res.append(bool(random.getrandbits(1)))
    return res
    
def neighbour_state_generator(state):
    new_state = state.copy()

    vertex_to_change = random.randint(0,len(new_state) - 1)
    previous_value = new_state[vertex_to_change]
    new_state[vertex_to_change] = not new_state[vertex_to_change]

    return new_state, previous_value, vertex_to_change

def cost_function(graph_matrix,state , A = 1 , B=1):

    cost = A * np.sum(state)
    for i in range(len(state)):
        for j in range(len(state)):
            cost += B * graph_matrix[i][j] * (not (state[i] or state[j]))
    return cost

deg = [np.sum(i) / edge_count for i in graph_matrix]

def prob_accept(current_state, next_state, temperature, index, graph_matrix, deg): 
    current_state_cost = cost_function(graph_matrix, current_state)
    next_state_cost = cost_function(graph_matrix, next_state)
    delta_f = next_state_cost - current_state_cost

    if current_state[index]:
        return exp(-(delta_f * (1 - deg[index]) / temperature))
    return exp(-(delta_f * (1 + deg[index]) / temperature))

def accept(current_state , next_state , graph_matrix, temperature ,index):
    delta_cost = cost_function(graph_matrix, next_state) - cost_function(graph_matrix, current_state)
    if delta_cost < 0:
        return True
    probablity = prob_accept(current_state,next_state, temperature, index, graph_matrix, deg)
    return random.random() < probablity

cost_list = []


def anneal(
    graph_matrix, stopping_temperature=1e-8, stopping_iter=2000, alpha=0.99, T=50
):
    cost_list.clear()
    n = len(graph_matrix)
    current_state = random_state_generator(n)
    best_solution = current_state
    best_cost = cost_function(graph_matrix, current_state)
    i = 0
    while True:
        if i > stopping_iter or T < stopping_temperature:
            break
        next_state, changed_value, index = neighbour_state_generator(current_state)
        if accept(current_state, next_state, graph_matrix, T, index):
            current_state = next_state
        current_cost = cost_function(graph_matrix, current_state)
        if  current_cost < best_cost:
            best_cost = current_cost
            best_solution = current_state
        T *= alpha
        i += 1
        cost_list.append(current_cost)
    
    
    return best_solution, best_cost

best_sol_SA, best_cost_SA = anneal(
    graph_matrix,
)
print(f"{best_sol_SA} with cost: {best_cost_SA}")

def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.show()   

plot_cost(cost_list)