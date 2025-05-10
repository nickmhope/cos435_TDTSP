import gymnasium
import gymnasium.spaces.discrete
import numpy as np

class CityEnv(gymnasium.Env):
  def __init__(self, N, time_horizon, poly_matrix, max_steps = 5_000):
    super().__init__()
    # environment parameters
    self.N = N
    self.time_horizon = time_horizon
    self.poly_matrix = poly_matrix
    self.max_steps = max_steps
    obs_dim = self.N + 2 # destination-encoding, current node, and current time
    self.action_space = gymnasium.spaces.Discrete(self.N)
    self.observation_space = gymnasium.spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # model-observable state
    # note that when the current_time is observed, the policy will actually be passed the time 
    # taken modulo self.time_horizon, since this is the relevant parameter when considering 
    # the temporal dynamics of the underlying environment
    self.current_time = 0
    self.destinations = None
    self.current_node = None

    # hidden state
    self.num_steps = 0
  
  # format state variables to be compatible with sb3
  def _format_obs(self):
    _time = np.array([self.current_time % self.time_horizon], dtype=np.float32)
    _destinations = self.destinations.astype(np.float32)
    _current_node = np.array([self.current_node], dtype=np.float32)
    return np.concatenate([_time, _destinations, _current_node])

  def reset(self, *, eval_params = None, seed = None):
    super().reset(seed=seed)
    self.destinations = np.random.choice([0, 1], size = self.N)
    self.current_time = np.random.uniform(0, 24)
    self.current_node = np.random.choice(self.N)
    self.num_steps = 0
    
    if eval_params is not None:
      self.destinations = eval_params['destinations']
      self.current_node = eval_params['current_node']
      self.current_time = eval_params['current_time']
    return self._format_obs(), {}
  
  def step(self, action):
    reward = 0.0
    if self.current_node == action:
        reward -= 1.0

    travel_time = self.poly_matrix[self.current_node, action].eval(self.current_time % self.time_horizon)
    self.current_time += travel_time

    if self.destinations[action] > 0:
        reward += 1.0
        if self.destinations.sum() == 1:
            reward += self.N
    self.destinations[action] = 0  # Mark destination as visited

    self.current_vertex = action # Move to current vertex
    self.num_steps += 1
    reward -= travel_time

    done = not self.destinations.any()
    truncated = self.num_steps >= self.max_steps

    return self._format_obs(), reward, done, truncated, {}

  def render(self):
    print(f"Time: {self.current_time}, Current node: {self.current_vertex}, Destinations: {self.destinations}")
    



    