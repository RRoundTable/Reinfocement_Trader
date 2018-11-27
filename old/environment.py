import tensorflow as tf
import random
from DeepQNet.readData import *

class Env:
    def __init__(self, normalized: np.ndarray, time_series: tuple, actions: np.ndarray,
                 input_size: int=50, transaction_cost = 0.0005):
        self.name = "env1"

        self.normalized = normalized
        self.price_diff = time_series[0]
        self.price = time_series[1]
        self.length = len(time_series[0])
        self.input_size = input_size
        self.index = 0
        self.actions = actions
        self.prev_action = 1
        self.tc = transaction_cost

    def state(self):
        state = self.normalized[self.index : self.index + self.input_size]
        price_diff = self.price_diff[self.index : self.index + self.input_size]
        price = self.price[self.index : self.index + self.input_size]
        return state, price_diff, price

    def reset(self):
        self.index = 0
        return self.state()[0]

    def step(self, action):
        self.index += 1
        done = False
        if self.index + self.input_size + 1 > self.length:
            done = True


        new_state, diff_, price_ = self.state()

        reward = diff_[-1] * self.actions[action]
        if self.actions[action] != self.actions[self.prev_action]:
            reward -= (self.tc * price_[-1]) * abs(self.actions[action] - self.actions[self.prev_action])
        self.prev_action = action

        return new_state, reward, done

    def action_sample(self) -> int:
        return np.random.randint(0, len(self.actions))


class Env2:
    def __init__(self, normalized: np.ndarray, time_series: tuple, actions: np.ndarray,
                 input_size: int=128, transaction_cost=0.0005):
        self.name = "env2"

        self.normalized = normalized
        self.price_diff = time_series[0]
        self.price = time_series[1]
        self.length = len(time_series[0])
        self.input_size = input_size
        self.index = 0
        self.actions = actions
        self.prev_action = 1
        self.tc = transaction_cost

    def state(self):
        state = np.append(self.normalized[self.index : self.index + self.input_size], self.actions[self.prev_action])
        price_diff = self.price_diff[self.index : self.index + self.input_size]
        price = self.price[self.index : self.index + self.input_size]
        return state, price_diff, price

    def reset(self):
        self.index = 0
        return self.state()[0]

    def step(self, action):
        self.index += 1
        done = False
        if self.index + self.input_size + 1 > self.length:
            done = True

        new_state, diff_, price_ = self.state()

        reward = diff_[-1] * self.actions[action]
        if self.actions[action] != self.actions[self.prev_action]:
            reward -= (self.tc * price_[-1]) * abs(self.actions[action] - self.actions[self.prev_action])
        self.prev_action = action

        return new_state, reward, done

    def action_sample(self) -> int:
        return np.random.randint(0, len(self.actions))


