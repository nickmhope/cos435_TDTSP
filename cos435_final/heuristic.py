
from poly_matrix import polynomial, create_poly_matrix
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

def discretize_poly_matrix(poly_matrix, N, n_intervals):
    disc_matrix = np.zeros((N, N, n_intervals))
    for i in range(N):
        for j in range(N):
            for k in range(n_intervals):
                t = k*24/n_intervals
                disc_matrix[i, j, t] = poly_matrix[i, j].eval(t)
    return disc_matrix 

def tsp_solver_dynamic_programming(travel_tensor, destinations, k):
    tensor = travel_tensor[:,:,:k].mean(axis=2).copy()
    keep = destinations.astype(bool)
    distance_matrix = tensor[keep][:, keep]
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    route = np.where(destinations == 1)[0][permutation]
    return route, distance

def tsp_heuristic_sim_annealing(travel_tensor, destinations, k):
    tensor = travel_tensor[:,:,:k].mean(axis=2).copy()
    keep = destinations.astype(bool)
    distance_matrix = tensor[keep][:, keep]
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix, max_processing_time=10)
    route = np.where(destinations == 1)[0][permutation]
    return route, distance

def verify_path(path, destinations):
    visit_nodes = {}
    for d in destinations:
        visit_nodes[d] = 0
    count = 0
    for v in path:
        if visit_nodes.get(v, -1) == 0:
            visit_nodes[v] = 1
            count += 1
    return count == len(destinations)
    
def evaluate_path(path, poly_matrix, start_time, start_node):
    i = start_node
    t = start_time
    for j in path[1:]:
        travel_time = poly_matrix[i, j].eval(t)
        t += travel_time
        i = j
    return t - start_time


def nearest_neighbors_instance(poly_matrix, start_time, start_node, destinations):
    destinations = list(np.where(np.array(destinations)==1)[0])
    t = start_time
    i = start_node
    D = len(destinations)
    path = []
    for k in range(D):
        times = np.array([poly_matrix[i][d].eval(t) for d in destinations])
        t += np.min(times)
        j = np.argmin(times)
        i = destinations.pop(j)
        path.append(i)
    return path

def test_nearest_neighbors(poly_matrix, start_times, start_nodes, destination_set):
    times = []
    for t, i, dests in zip(start_times, start_nodes, destination_set):
        path = nearest_neighbors_instance(poly_matrix, t, i, dests)
        verify_path(path, dests)
        times.append(evaluate_path(path, poly_matrix, t, i))
    return np.mean(times), times


def test_dynamic_programming_tsp(poly_matrix, start_times, start_nodes, destination_set):
    times = []
    for t, i, dests in zip(start_times, start_nodes, destination_set):
        path = tsp_solver_dynamic_programming(poly_matrix, t, i, dests)
        assert(verify_path(path, dests))
        times.append(evaluate_path(path, poly_matrix, t, i))
    return np.mean(times), times


if __name__ == '__main__':
    poly_matrix = create_poly_matrix(N=20, time_horizon=24)
    destination_set = []
    start_nodes = []
    start_times = []
    for i in range(10000):
        vector = np.random.randint(0, 2, size=20)
        destination_set.append(list(vector))
        start_nodes.append(0)
        start_times.append(int(np.random.choice(list(range(24)))))
    start_times = list(start_times)
    start_nodes = list(start_nodes)
    avg, times =test_nearest_neighbors(poly_matrix=poly_matrix, start_times=start_times, start_nodes=start_nodes, destination_set=destination_set)
    print(avg)

        





