# @Author  : Zheng Li
# @Time    : 2018/7/23
# @Version : 1.0
# @Document: Reinforcement-learning-with-tensorflow-master.ipynb


import time
import numpy as np
import pandas as pd


color_bar = {
    2: '#6495ED',
    4: '#1E90FF',
    8: '#008B8B',
    16: '#FF8C00',
    32: '#B8860B',
    64: '#FF7F50',
    128: '#DC143C',
    256: '#006400',
    512: '#9932CC',
    1024: '#FF1493',
    2048: '#C71585',
    4096: '#800080',
    8192: '#191970',
    16384: '#A52A2A'
}


def build_grid(WIDTH, MODE, SPACING, canvas):
    width = (WIDTH - (MODE + 1) * SPACING) / MODE
    theta_1 = np.linspace(0, np.pi/2, num=11, endpoint=True)
    edge_1 = SPACING * np.hstack((np.cos(theta_1).reshape(-1, 1), np.sin(theta_1).reshape(-1, 1)))
    edge_1 += np.array([width / 2 - SPACING, width / 2 - SPACING])

    theta_2 = np.linspace(np.pi/2, np.pi, num=11, endpoint=True)
    edge_2 = SPACING * np.hstack((np.cos(theta_2).reshape(-1, 1), np.sin(theta_2).reshape(-1, 1)))
    edge_2 += np.array([-width / 2 + SPACING, width / 2 - SPACING])

    theta_3 = np.linspace(np.pi, np.pi*3/2, num=11, endpoint=True)
    edge_3 = SPACING * np.hstack((np.cos(theta_3).reshape(-1, 1), np.sin(theta_3).reshape(-1, 1)))
    edge_3 += np.array([-width / 2 + SPACING, -width / 2 + SPACING])

    theta_4 = np.linspace(np.pi*3/2, np.pi*2, num=11, endpoint=True)
    edge_4 = SPACING * np.hstack((np.cos(theta_4).reshape(-1, 1), np.sin(theta_4).reshape(-1, 1)))
    edge_4 += np.array([width / 2 - SPACING, -width / 2 + SPACING])

    edge = np.vstack((edge_1, edge_2, edge_3, edge_4))

    grid = np.linspace(-width / 2, WIDTH + SPACING / 2 + width / 2, num=MODE + 2, endpoint=True)
    grid = {key: value for key, value in zip(list(range(-1, MODE + 2)), grid)}

    for i in range(MODE):
        for j in range(MODE):
            rect = (edge + np.array([grid[i], grid[j]])).reshape(-1)
            canvas.create_polygon(*rect, fill='DarkGray')

    unit = width + SPACING
    return grid, unit, edge


class SaveData:
    def __init__(self, MODE):
        self.MODE = MODE

        init_ids = np.random.choice(MODE, 4, replace=False)
        while init_ids[0] == init_ids[2] and init_ids[1] == init_ids[3]:
            init_ids = np.random.choice(MODE, 4, replace=False) + 1
        self.state = pd.DataFrame(
            np.zeros((self.MODE + 2, self.MODE + 2), dtype=np.int64),
            columns=list(range(-1, MODE + 1)),
            index=list(range(-1, MODE + 1))
        )
        self.state.loc[init_ids[0], init_ids[1]] = 2
        self.state.loc[init_ids[2], init_ids[3]] = 2

        self.collection = {}

    def refresh_canvas(self, canvas, edge, grid):
        for value in self.collection.values():
            canvas.delete(value[0])
            canvas.delete(value[1])
        self.collection = {}
        for i in range(-1, self.MODE+1):
            for j in range(-1, self.MODE+1):
                if self.state.loc[i, j] != 0:
                    rect = (edge + np.array([grid[j], grid[i]])).reshape(-1)
                    element = canvas.create_polygon(
                        *rect, fill=color_bar[self.state.loc[i, j]], tag='{},{}'.format(i, j))
                    text = canvas.create_text(
                        grid[j], grid[i], text=str(self.state.loc[i, j]), fill='White', tag='{}.{}'.format(i, j))
                    self.collection[(i, j)] = element, text

    def table_get(self, direct, state):
        if direct == 'top':
            return np.rot90(state.iloc[1:self.MODE + 1, 1:self.MODE + 1].values, k=0)
        elif direct == 'left':
            return np.rot90(state.iloc[1:self.MODE + 1, 1:self.MODE + 1].values, k=-1)
        elif direct == 'bottom':
            return np.rot90(state.iloc[1:self.MODE + 1, 1:self.MODE + 1].values, k=-2)
        elif direct == 'right':
            return np.rot90(state.iloc[1:self.MODE + 1, 1:self.MODE + 1].values, k=-3)
        else:
            raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')

    def state_set(self, direct, state, table):
        if direct == 'top':
            state.iloc[1:self.MODE + 1, 1:self.MODE + 1] = np.rot90(table, k=0)
        elif direct == 'left':
            state.iloc[1:self.MODE + 1, 1:self.MODE + 1] = np.rot90(table, k=1)
        elif direct == 'bottom':
            state.iloc[1:self.MODE + 1, 1:self.MODE + 1] = np.rot90(table, k=2)
        elif direct == 'right':
            state.iloc[1:self.MODE + 1, 1:self.MODE + 1] = np.rot90(table, k=3)
        else:
            raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')
        return state

    def move_map_element(self, move_map, j, i, cursor, direct):
        if direct == 'top':
            move_map[(i, j)] = (cursor, j)
        elif direct == 'left':
            move_map[(self.MODE - 1 - j, i)] = (self.MODE - 1 - j, cursor)
        elif direct == 'bottom':
            move_map[(self.MODE - 1 - i, self.MODE - 1 - j)] = (self.MODE - 1 - cursor, self.MODE - 1 - j)
        elif direct == 'right':
            move_map[(j, self.MODE - 1 - i)] = (j, self.MODE - 1 - cursor)
        else:
            raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')
        return move_map

    def predict_state(self, direct, input_state):
        """
        Note: this function only used to predict next state. In fact won't update state.
        input_state -> state -> next_state
                    ^        ^
                    |        |
                 random  move_map
        """
        table = self.table_get(direct, input_state)
        next_table = np.zeros_like(table)
        move_map = {}

        for j in range(self.MODE):
            cursor = 0
            for i in range(self.MODE):
                if table[i, j] == 0:
                    continue
                else:
                    if cursor == next_table[cursor, j] == 0:
                        next_table[cursor, j] = table[i, j]
                        self.move_map_element(move_map, j, i, cursor, direct)
                    else:
                        if next_table[cursor, j] != 0:
                            if table[i, j] == next_table[cursor, j]:
                                next_table[cursor, j] *= 2
                                self.move_map_element(move_map, j, i, cursor, direct)
                                cursor += 1
                            else:
                                cursor += 1
                                next_table[cursor, j] = table[i, j]
                                self.move_map_element(move_map, j, i, cursor, direct)
                        elif table[i, j] != 0:
                            next_table[cursor, j] = table[i, j]
                            self.move_map_element(move_map, j, i, cursor, direct)

        state = self.state_set(direct, input_state, table)

        collection = []
        for j in range(self.MODE):
            if next_table[-1, j] == 0:
                collection.append(j)
        if collection:
            ind = np.random.choice(np.array(collection), 1)[0]
            num = np.random.choice(np.array([2, 4]), 1)
            next_table[-1, ind] = num
            self.move_map_element(move_map, ind, self.MODE, self.MODE - 1, direct)

            if direct == 'top':
                state.loc[self.MODE, ind] = num
            elif direct == 'left':
                state.loc[self.MODE - 1 - ind, self.MODE] = num
            elif direct == 'bottom':
                state.loc[-1, self.MODE - 1 - ind] = num
            elif direct == 'right':
                state.loc[ind, -1] = num
            else:
                raise ValueError('"direct" should in ["top", "left", "bottom", "right"]')

        next_state = pd.DataFrame(
            np.zeros((self.MODE+2, self.MODE+2), dtype=np.int64), columns=list(range(-1, self.MODE+1)),
            index=list(range(-1, self.MODE+1)))
        next_state.iloc[1:self.MODE + 1, 1:self.MODE + 1] = next_table

        next_state = self.state_set(direct, next_state, next_table)

        return state, next_state, move_map

    def update_canvas(self, FRAMES, PAUSE, canvas, edge, grid, direct):
        state, next_state, move_map = self.predict_state(direct, self.state)
        self.state = state
        self.refresh_canvas(canvas, edge, grid)

        move_message = {}
        for key, value in move_map.items():
            d_i = (grid[value[0]] - grid[key[0]]) / FRAMES
            d_j = (grid[value[1]] - grid[key[1]]) / FRAMES
            move_message['{},{}'.format(key[0], key[1])] = (d_i, d_j)

        for i in range(FRAMES):
            for key, value in move_message.items():
                canvas.move(key, value[1], value[0])
                canvas.move(key.replace(',', '.'), value[1], value[0])
            canvas.update()
            if i != FRAMES - 1:
                time.sleep(PAUSE)

        self.state = next_state
        self.refresh_canvas(canvas, edge, grid)


def build_canvas(WIDTH, MODE, SPACING, FRAMES, PAUSE, canvas):
    grid, unit, edge = build_grid(WIDTH, MODE, SPACING, canvas)
    data = SaveData(MODE)
    data.refresh_canvas(canvas, edge, grid)

    def merge(direct):
        data.update_canvas(FRAMES, PAUSE, canvas, edge, grid, direct)

    return grid, unit, merge
