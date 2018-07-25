# @Author  : Zheng Li
# @Time    : 2018/7/25
# @Version : 1.0
# @Document: RL2048.ipynb
import pickle
import numpy as np


def reward(state):
    table = state.copy().values[1:-1, 1:-1]

    score = np.sum((table // 4) ** 2)

    zero = np.sum(table == 0) * np.mean(table)

    dev = np.cov(table.reshape(-1))

    table = np.flipud(table)
    for i in range(table.__len__()):
        if i % 2 == 1:
            table[i] = table[i, ::-1]
    table = np.array([it for it in table.reshape(-1) if it != 0])
    table = table[1:] * table[:-1] * (2 * (table[1:] <= table[:-1]) - 1)
    seq = np.sum(table)

    return score, zero, dev, seq


def choose_activate(data):
    state, next_state, move_map = data.predict_state(data.MODE, 'top', data.state.copy())
    r_up = np.sum(reward(next_state))

    state, next_state, move_map = data.predict_state(data.MODE, 'left', data.state.copy())
    r_left = np.sum(reward(next_state))

    state, next_state, move_map = data.predict_state(data.MODE, 'bottom', data.state.copy())
    r_bottom = np.sum(reward(next_state))

    state, next_state, move_map = data.predict_state(data.MODE, 'right', data.state.copy())
    r_right = np.sum(reward(next_state))

    ind = np.argmax([r_up, r_left, r_bottom, r_right])

    actions = ['top', 'left', 'bottom', 'right']

    return actions[ind]
