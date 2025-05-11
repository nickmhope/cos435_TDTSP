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
        total_time += (eval_env.current_time - start_time_set[i]) / benchmark_size
        avg_reward = total_reward / benchmark_size

    return total_time, avg_reward

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
        rows = [[i, time, avg_reward] for i, (time, avg_reward) in enumerate(zip(benchmark_average_travel_times, benchmark_average_rewards))]
        pd.DataFrame(rows, columns=['step', 'average_time', 'average_reward']).to_csv(f'{savefile_name}.csv')