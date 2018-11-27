import numpy as np
from sklearn.preprocessing import normalize


class Env:
    def __init__(self, num_data, raw_data, end_data, actions, input_size, seq_size, name, transaction_cost=0.005):
        idx_range = range(num_data[0], num_data[1])
        self.length = len(idx_range)                              # N
        self.norm_data = normalize(raw_data[idx_range], axis=0)   # [N, 13] normalize each feature
        self.end_data = end_data[idx_range]                       # [N,]
        self.end_diff = self.end_data[1:] - self.end_data[:-1]    # [N-1,]
        self.end_diff = np.append(self.end_diff, self.end_diff[-1])  # [N,] fill the last value double
        self.actions = actions                                    # [3,]
        self.input_size = input_size                              # 13
        self.seq_size = seq_size                                  # S
        self.name = name
        self.tc = transaction_cost

        self.prev_action = 1
        self.index = 0

    def state(self):
        idx_range = range(self.index - (self.seq_size - 1), self.index + 1)

        if self.index + 1 == self.length:  # isTerminalState
            done = True
        else:
            done = False

        end_diff = self.end_diff[idx_range]  # [S,]
        state = self.norm_data[idx_range]    # [S, 13]
        price = self.end_data[idx_range]     # [S,]

        return state, end_diff, price, done

    def reset(self):
        self.index = self.seq_size - 1  # 0

        initial_state, _, _, _ = self.state()
        return initial_state

    def step(self, action):
        next_state, diff, price, done = self.state()
        reward = diff[-1] * self.actions[action]  # -1: Sell, 0: Hold, 1: Buy

        # Not Used
        # if cur action is different from prev action
        # if action != self.prev_action:  # action = 매도 / 매수의 행위가 아니라 position
        #     reward -= (self.tc * price[-1]) * abs(self.actions[action] - self.actions[self.prev_action])
        # self.prev_action = action

        self.index += 1

        return next_state, reward, done

    def action_sample(self):
        return np.random.randint(0, len(self.actions))
