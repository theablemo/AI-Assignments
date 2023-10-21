student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'



def solve(N, M, K, NUMS, roads): 
    pass

def manhattan_dist(x1, x2, y1, y2):
    return abs(x2 - x1) + abs (y2 - y1)
def euclidean_dist(x1, x2, y1, y2):
    x_diff = abs(x2 - x1)
    y_diff = abs(y2 - y1)
    return math.sqrt(pow(x_diff, 2) + pow(y_diff, 2))

def heur_displaced(state):    
    boxes = state.boxes
    storage = state.storage
    displaced_boxes = 0
    in_storage = False
    for coordinate_box in boxes:
        in_storage = False
        for coordinate_storage in storage:
            if coordinate_box == coordinate_storage:
                in_storage = True
                break
        if not in_storage:
            displaced_boxes += 1
    
    return displaced_boxes

def heur_manhattan_distance(state):
    restrictions = state.restrictions
    boxes = state.boxes
    storage = state.storage
    shortest_distances = 0
    if restrictions == None:
        for coordinate_box in boxes:
            x_box = coordinate_box[0]
            y_box = coordinate_box[1]
            box_closest_distance = math.inf
            for coordinate_storage in storage:
                x_storage = coordinate_storage[0]
                y_storage = coordinate_storage[1]
                distance = manhattan_dist(x_box,x_storage,y_box,y_storage)
                if distance < box_closest_distance:
                    box_closest_distance = distance
            shortest_distances += box_closest_distance
    else:
        for coordinate_box in boxes:
            box_index = boxes[coordinate_box]
            storage_list = list(restrictions[box_index])
            x_box = coordinate_box[0]
            y_box = coordinate_box[1]
            box_closest_distance = math.inf
            for coordinate_storage in storage_list:
                x_storage = coordinate_storage[0]
                y_storage = coordinate_storage[1]
                distance = manhattan_dist(x_box, x_storage, y_box, y_storage)
                if distance < box_closest_distance:
                    box_closest_distance = distance
            shortest_distances += box_closest_distance
    return shortest_distances

def heur_euclidean_distance(state):  

    restrictions = state.restrictions
    boxes = state.boxes
    storage = state.storage
    shortest_distances = 0
    if restrictions == None:
        for coordinate_box in boxes:
            x_box = coordinate_box[0]
            y_box = coordinate_box[1]
            box_closest_distance = math.inf
            for coordinate_storage in storage:
                x_storage = coordinate_storage[0]
                y_storage = coordinate_storage[1]
                distance = euclidean_dist(x_box,x_storage,y_box,y_storage)
                if distance < box_closest_distance:
                    box_closest_distance = distance
            shortest_distances += box_closest_distance
    else:
        for coordinate_box in boxes:
            box_index = boxes[coordinate_box]
            storage_list = list(restrictions[box_index])
            x_box = coordinate_box[0]
            y_box = coordinate_box[1]
            box_closest_distance = math.inf
            for coordinate_storage in storage_list:
                x_storage = coordinate_storage[0]
                y_storage = coordinate_storage[1]
                distance = euclidean_dist(x_box, x_storage, y_box, y_storage)
                if distance < box_closest_distance:
                    box_closest_distance = distance
            shortest_distances += box_closest_distance
    return shortest_distances


plt.plot(displace_time, 'r--', manhattan_time, 'bs', euclidean_time, 'g^')
plt.ylabel("Search Time")
plt.xlabel("Problem")
plt.show()

plt.plot(displace_nodes, 'r--', manhattan_node, 'bs', euclidean_nodes, 'g^')
plt.ylabel("Nodes Expanded")
plt.xlabel("Problem")
plt.show()

plt.plot(displace_states, 'r--', manhattan_states, 'bs', euclidean_states, 'g^')
plt.ylabel("States Generated")
plt.xlabel("Problem")
plt.show()

problems_nodes_expanded = []
def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10, problem_index=1):
    best_path_cost = float("inf")
    time_remain = 8
    iter = 0

    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)
    time_started = time.time()
    while (time_remain > 0) and not se.open.empty():
        final, nodes_expanded = se.search(timebound, (float("inf"),float("inf"),best_path_cost))
        if final:
            final_path_cost = final.gval + heur_fn(final)
            problems_nodes_expanded[problem_index].append(nodes_expanded)
            if final_path_cost < best_path_cost:
                best_path_cost = final_path_cost
                optimal_final = final
       
        time_passed = time.time() - time_started
        time_remain -= time_passed
        iter +=1
    try:
        return optimal_final, nodes_expanded
    except:
        return final, nodes_expanded

    return False


i = 0
for problem in problems_nodes_expanded:
    plt.plot(problem, 'ro')
    plt.ylabel(f"Nodes Expanded For Each Goal In Problem {i}")
    plt.xlabel("#Iteration")
    plt.show()
    i += 1



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

def plot_cost(cost_list):
    
    plt.plot(cost_list)
    plt.show()   

plot_cost(cost_list)

def individual_generator(n):
    res = []
    for _ in range(n):
        res.append(bool(random.getrandbits(1)))
    return res
def not_covered_edges(graph, state):
    n = len(graph)
    not_covered_edges_in = 0
    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1 and not (state[i] or state[j]):
                not_covered_edges_in += 1
    not_covered_edges_in = not_covered_edges_in / 2
    return not_covered_edges_in

def population_generation(n, k): 
   
    population = []
    for _ in range(k):
        population.append(individual_generator(n))
    return population

def cost_function2(graph,state):
    
    number_of_vertices_in_state = np.sum(state)
    number_of_not_covered_edges = not_covered_edges(graph, state)
    cost = int(1 * number_of_vertices_in_state + 5 * number_of_not_covered_edges)
    return cost



def tournament_selection(graph, population):
    
    new_population = []
    population_size = len(population)
    for _ in population:
        random_index_one = random.randint(0,population_size - 1)
        random_index_two = random.randint(0, population_size - 1)
        if cost_function2(graph, population[random_index_one]) < cost_function2(graph, population[random_index_two]):
            new_population.append(population[random_index_one])
        else:
            new_population.append(population[random_index_two])
    return new_population
    

def crossover(graph, parent1, parent2):
    
    parent_size = len(parent1)
    random_index = random.randint(0,parent_size)
    child1 = parent1[:random_index] + parent2[random_index:]
    child2 = parent2[:random_index] + parent1[random_index:]
    return child1, child2

def mutation(graph,chromosme,probability):
    
    random_number = random.random()
    if random_number > probability:
        chromosme_size = len(chromosme)
        random_index = random.randint(0, chromosme_size - 1)
        chromosme[random_index] = not chromosme[random_index]
    

def genetic_algorithm(graph_matrix,mutation_probability=0.1,pop_size=100,max_generation=100):
    
    n = len(graph_matrix)
    population = population_generation(n, pop_size)
    best_cost = math.inf
    best_solution = None
    current_generation = 1
    while current_generation <= max_generation:
        population = tournament_selection(graph_matrix,population)
        crossover_index = 0
        while crossover_index < len(population) - 1:
            parent1 = population[crossover_index]
            one_index = crossover_index
            crossover_index += 1
            parent2 = population[crossover_index]
            two_index = crossover_index
            crossover_index += 1
            population[one_index], population[two_index] = crossover(graph_matrix, parent1, parent2)
        for chromosome in population:
            mutation(graph_matrix, chromosome, mutation_probability)
        for state in population:
            state_cost = cost_function2(graph_matrix, state)
            if state_cost < best_cost:
                best_cost = state_cost
                best_solution = state
        current_generation += 1
    return best_cost,best_solution

