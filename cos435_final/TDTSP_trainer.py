import numpy as np
from poly_matrix import polynomial, create_poly_matrix
from city_env import CityEnv  # make sure the env class is renamed properly if needed
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

# define helper function for model evaluation
def eval_func(model):
    eval_env = CityEnv(poly_matrix=poly_matrix, N=N, time_horizon=T)
    total_time = 0
    for i in range(benchmark_size):
        benchmark_element = {
        "destinations": destination_set[i].copy(),
        "current_time": start_time_set[i],
        "current_node": start_node_set[i] }
        obs, _ = eval_env.reset(eval_params=benchmark_element or {})
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            #print(obs)
            action, _ = model.predict(obs, deterministic=True)
            #print(action)
            obs, reward, done, truncated, _ = eval_env.step(action)

            total_reward += reward
            if truncated and not done:
                print(f'WARNING: TRUNCATION on benchmark example {i}')
        total_time += eval_env.current_time / benchmark_size
        avg_reward = total_reward / benchmark_size

    return total_time, avg_reward

def discretize_poly_matrix(poly_matrix, N):
    discretization_constant = 1000
    tensor = np.zeros((N, N, discretization_constant), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            for k in range(discretization_constant):
                t = k * 0.024
                tensor[i, j, k] = poly_matrix[i, j].eval(t)
    return tensor

def nearest_neighbors_with_time(travel_tensor, start_node, visit_nodes):
    visit_nodes = visit_nodes.copy()
    path = [start_node]
    current_node = start_node
    current_time = 0
    total_time = 0

    while len(visit_nodes) > 0:
        best_node = None
        best_time = float('inf')
        for node in visit_nodes:
            time_index = min(int(current_time), travel_tensor.shape[2] - 1)
            travel_time = travel_tensor[current_node, node, time_index]
            if travel_time < best_time:
                best_time = travel_time
                best_node = node

        if best_node is None:
            break

        # Update current state
        visit_nodes.remove(best_node)
        current_time += best_time
        total_time += best_time
        path.append(best_node)
        current_node = best_node

    return path, total_time

def tsp_solver_dynamic_programming(travel_tensor, destinations):
    tensor = travel_tensor[:,:,:60].mean(axis=2).copy()
    keep = destinations.astype(bool)
    distance_matrix = tensor[keep][:, keep]
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    route = np.where(destinations == 1)[0][permutation]
    return route, distance

def tsp_heuristic_sim_annealing(travel_tensor, destinations):
    tensor = travel_tensor[:,:,:60].mean(axis=2).copy()
    keep = destinations.astype(bool)
    distance_matrix = tensor[keep][:, keep]
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix, max_processing_time=180)
    route = np.where(destinations == 1)[0][permutation]
    return route, distance

def evaluate_tsp_solver(permutation, poly_matrix):
    i = permutation[0]
    t = 0
    for j in permutation[1:]:
        travel_time = poly_matrix[i, j].eval(t)
        t += travel_time
        i = j
    return t
    visit_nodes = visit_nodes.copy()
    path = [start_node]
    current_node = start_node
    current_time = 0
    total_time = 0

    while len(visit_nodes) > 0:
        best_node = None
        best_time = float('inf')
        for node in visit_nodes:
            time_index = min(int(current_time), travel_tensor.shape[2] - 1)
            travel_time = travel_tensor[current_node, node, time_index]
            if travel_time < best_time:
                best_time = travel_time
                best_node = node

        if best_node is None:
            break

        # Update current state
        visit_nodes.remove(best_node)
        current_time += best_time
        total_time += best_time
        path.append(best_node)
        current_node = best_node

    return path, total_time

if __name__ == '__main__':
    # Environment parameters
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
        "N",
        type=int,
        default=10,
        )
    CLI.add_argument(
        "num_iters",
        type=int,
        default=300,
        )
    CLI.add_argument(
        "train_steps",
        type=int,
        default=500_000,
        )
    CLI.add_argument(
        "benchmark_size",
        type=int,
        default=100,
        )
    args = CLI.parse_args()

    N = args.N
    num_iters = args.num_iters
    train_steps = args.train_steps
    benchmark_size = args.benchmark_size
    NUM_ENVIRONMENTS = 50

    # Initialize environment
    T = 24
    poly_matrix = create_poly_matrix(N, T)

    def make_env():
        return CityEnv(poly_matrix=poly_matrix, N=N, time_horizon=T)

    env = make_vec_env(make_env, n_envs=NUM_ENVIRONMENTS)

    # Initialize benchmark set and savefile for model
    destination_set = [np.random.choice([0, 1], size=N) for i in range(benchmark_size)]
    start_node_set = np.random.choice(np.arange(N), size = benchmark_size)
    start_time_set = np.random.uniform(low = 0, high = 24, size = benchmark_size)
    savefile_name = f'evaluations/PPO_N={N}_iters={num_iters}'

    # create and save model
    model = PPO("MlpPolicy", env, verbose=0)
    model.save(savefile_name)

    with open(f"{savefile_name}_destinations_set.pkl", "wb") as f:
        pickle.dump(destination_set, f)
    with open(f"{savefile_name}_start_node_set.pkl", "wb") as f:
        pickle.dump(start_node_set, f)
    with open(f"{savefile_name}_start_time_set.pkl", "wb") as f:
        pickle.dump(start_time_set, f)
    with open(f"{savefile_name}_poly_matrix.pkl", "wb") as f:
        pickle.dump(poly_matrix, f)
    with open(f"{savefile_name}_heuristic_stats.txt", 'w') as f:
        f.write('Heuristics:\n')

    travel_tensor = discretize_poly_matrix(poly_matrix, N)
    total_time = 0
    for destinations in destination_set:
        visit_nodes = np.where(destinations == 1)[0]
        visit_nodes = visit_nodes.tolist()
        start_node = 0
        path, travel_time = nearest_neighbors_with_time(travel_tensor, start_node, visit_nodes)
        total_time += travel_time
    with open(f"{savefile_name}_heuristic_stats.txt", 'a') as f:
        f.write(f'NN: {total_time}\n')

    distances = []
    for destinations in destination_set:
        permutation, distance = tsp_heuristic_sim_annealing(travel_tensor, destinations)
        distances.append(evaluate_tsp_solver(permutation, poly_matrix))

    with open(f"{savefile_name}_heuristic_stats.txt", 'a') as f:
        f.write(f'sim_annealing_TSP_Heuristic: {np.mean(distances)}\n')

    benchmark_average_travel_times = []
    benchmark_average_rewards = []

    for i in range(num_iters):
        print('Training epoch: ', i)
        model = PPO.load(savefile_name, env=env)
        model.learn(total_timesteps=train_steps)

        benchmark_time, average_reward = eval_func(model)
        print('Benchmark time', benchmark_time, 'Average reward', average_reward)
        benchmark_average_travel_times.append(benchmark_time)
        benchmark_average_rewards.append(average_reward)

        model.save(savefile_name)
        rows = [[i, time, avg_reward] for i, time, avg_reward in enumerate(zip(benchmark_average_travel_times, benchmark_average_rewards))]
        pd.DataFrame(rows, columns=['step', 'average_time', 'average_reward']).to_csv(f'{savefile_name}.csv')