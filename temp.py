# @Author  : Zheng Li
# @Time    : 2018/7/23
# @Version : 1.0
# @Document: Reinforcement-learning-with-tensorflow-master.ipynb


import numpy as np
import pandas as pd


def init_demo():
    MODE = 12
    direct = 'right'  # 'top', 'left', 'bottom', 'right'

    state = pd.DataFrame(
        np.zeros((MODE+2, MODE+2), dtype=np.int8), columns=list(range(-1, MODE+1)), index=list(range(-1, MODE+1)))

    table = np.zeros((MODE, MODE), dtype=np.int8)
    table[:, 0] = np.array([3, 3, 0, 2, 2, 0, 0, 3, 5, 5, 3, 3])
    for j in range(1, MODE):
        table[:, j] = table[:, 0]
        if j > MODE / 2:
            table[:, j] += np.array(list(range(1, 2 * MODE + 1, 2)))
        np.random.shuffle(table[:, j])

    if direct == 'top':
        state.iloc[1:MODE+1, 1:MODE+1] = table
    elif direct == 'left':
        state.iloc[1:MODE+1, 1:MODE+1] = np.rot90(table, k=1)
    elif direct == 'bottom':
        state.iloc[1:MODE+1, 1:MODE+1] = np.rot90(table, k=2)
    elif direct == 'right':
        state.iloc[1:MODE+1, 1:MODE+1] = np.rot90(table, k=3)
    else:
        raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')
    return MODE, direct, state


def table_get(MODE, direct, state):
    if direct == 'top':
        return np.rot90(state.iloc[1:MODE+1, 1:MODE+1].values, k=0)
    elif direct == 'left':
        return np.rot90(state.iloc[1:MODE+1, 1:MODE+1].values, k=-1)
    elif direct == 'bottom':
        return np.rot90(state.iloc[1:MODE+1, 1:MODE+1].values, k=-2)
    elif direct == 'right':
        return np.rot90(state.iloc[1:MODE+1, 1:MODE+1].values, k=-3)
    else:
        raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')


def state_set(MODE, direct, state, table):
    if direct == 'top':
        state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=0)
    elif direct == 'left':
        state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=1)
    elif direct == 'bottom':
        state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=2)
    elif direct == 'right':
        state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=3)
    else:
        raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')
    return state


def move_map_element(MODE, move_map, j, i, cursor, direct):
    if direct == 'top':
        move_map[(i, j)] = (cursor, j)
    elif direct == 'left':
        move_map[(MODE - 1 - j, i)] = (MODE - 1 - j, cursor)
    elif direct == 'bottom':
        move_map[(MODE - 1 - i, MODE - 1 - j)] = (MODE - 1 - cursor, MODE - 1 - j)
    elif direct == 'right':
        move_map[(j, MODE - 1 - i)] = (j, MODE - 1 - cursor)
    else:
        raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')
    return move_map


def update_state(MODE, direct, state):
    table = table_get(MODE, direct, state)
    next_table = np.zeros_like(table)
    move_map = {}

    for j in range(MODE):
        cursor = 0
        for i in range(MODE):
            if table[i, j] == 0:
                continue
            else:
                if cursor == next_table[cursor, j] == 0:
                    next_table[cursor, j] = table[i, j]
                    move_map_element(MODE, move_map, j, i, cursor, direct)
                else:
                    if next_table[cursor, j] != 0:
                        if table[i, j] == next_table[cursor, j]:
                            next_table[cursor, j] *= 2  # TODO: add reword here!
                            move_map_element(MODE, move_map, j, i, cursor, direct)
                            cursor += 1
                        else:
                            cursor += 1
                            next_table[cursor, j] = table[i, j]
                            move_map_element(MODE, move_map, j, i, cursor, direct)
                    elif table[i, j] != 0:
                        next_table[cursor, j] = table[i, j]
                        move_map_element(MODE, move_map, j, i, cursor, direct)

    collection = []
    for j in range(MODE):
        if next_table[-1, j] == 0:
            collection.append(j)
    if collection:
        ind = np.random.choice(np.array(collection), 1)[0]
        next_table[-1, ind] = 2
        move_map_element(MODE, move_map, ind, MODE+1, MODE, direct)

        if direct == 'top':
            state.loc[MODE, ind] = 2
        elif direct == 'left':
            state.loc[MODE - 1 - ind, MODE] = 2
        elif direct == 'bottom':
            state.loc[-1, MODE - 1 - ind] = 2
        elif direct == 'right':
            state.loc[ind, -1] = 2
        else:
            raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')

    state = state_set(MODE, direct, state, table)

    next_state = pd.DataFrame(
        np.zeros((MODE+2, MODE+2), dtype=np.int8), columns=list(range(-1, MODE+1)), index=list(range(-1, MODE+1)))
    next_state.iloc[1:MODE+1, 1:MODE+1] = next_table

    next_state = state_set(MODE, direct, next_state, next_table)

    print('state = \n', state)
    print('next_state = \n', next_state)
    print(move_map)
    return state, next_state, move_map


def refresh(state):
    pass


MODE, direct, state = init_demo()
update_state(MODE, direct, state)
