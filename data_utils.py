# @Author  : Zheng Li
# @Time    : 2018/7/23
# @Version : 1.0
# @Document: RL2048.ipynb
import time
import pickle
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


class SaveData:
    def __init__(self, WIDTH, MODE, SPACING, FRAMES, PAUSE):
        self.MODE = MODE
        self.edge, self.grid, self.unit = self.build_units(WIDTH, MODE, SPACING)
        self.FRAMES = FRAMES
        self.PAUSE = PAUSE

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

    @staticmethod
    def build_units(WIDTH, MODE, SPACING):
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

        unit = width + SPACING
        return edge, grid, unit

    @classmethod
    def table_get(cls, MODE, action, state):
        if action == 'top':
            return np.rot90(state.iloc[1:MODE + 1, 1:MODE + 1].values, k=0)
        elif action == 'left':
            return np.rot90(state.iloc[1:MODE + 1, 1:MODE + 1].values, k=-1)
        elif action == 'bottom':
            return np.rot90(state.iloc[1:MODE + 1, 1:MODE + 1].values, k=-2)
        elif action == 'right':
            return np.rot90(state.iloc[1:MODE + 1, 1:MODE + 1].values, k=-3)
        else:
            raise ValueError('"action" should in ["top", "left", "bottom", "right"]')

    @classmethod
    def state_set(cls, MODE, action, state, table):
        if action == 'top':
            state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=0)
        elif action == 'left':
            state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=1)
        elif action == 'bottom':
            state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=2)
        elif action == 'right':
            state.iloc[1:MODE + 1, 1:MODE + 1] = np.rot90(table, k=3)
        else:
            raise ValueError('"action" should in ["top", "left", "bottom", "right"]')
        return state

    @classmethod
    def move_map_element(cls, MODE, move_map, j, i, cursor, action):
        if action == 'top':
            move_map[(i, j)] = (cursor, j)
        elif action == 'left':
            move_map[(MODE - 1 - j, i)] = (MODE - 1 - j, cursor)
        elif action == 'bottom':
            move_map[(MODE - 1 - i, MODE - 1 - j)] = (MODE - 1 - cursor, MODE - 1 - j)
        elif action == 'right':
            move_map[(j, MODE - 1 - i)] = (j, MODE - 1 - cursor)
        else:
            raise ValueError('"action" should in ["top", "left", "bottom", "right"]')
        return move_map

    @classmethod
    def predict_state(cls, MODE, action, input_state):
        """
        Note: this function only used to predict next state. In fact won't update state.
        input_state -> state -> next_state
                    ^        ^
                    |        |
                 random  move_map
        """
        table = cls.table_get(MODE, action, input_state)
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
                        cls.move_map_element(MODE, move_map, j, i, cursor, action)
                    else:
                        if next_table[cursor, j] != 0:
                            if table[i, j] == next_table[cursor, j]:
                                next_table[cursor, j] *= 2
                                cls.move_map_element(MODE, move_map, j, i, cursor, action)
                                cursor += 1
                            else:
                                cursor += 1
                                next_table[cursor, j] = table[i, j]
                                cls.move_map_element(MODE, move_map, j, i, cursor, action)
                        elif table[i, j] != 0:
                            next_table[cursor, j] = table[i, j]
                            cls.move_map_element(MODE, move_map, j, i, cursor, action)

        state = cls.state_set(MODE, action, input_state, table)

        collection = []
        for j in range(MODE):
            if next_table[-1, j] == 0:
                collection.append(j)
        if collection:
            ind = np.random.choice(np.array(collection), 1)[0]
            num = np.random.choice(np.array([2, 4]), 1, p=[0.7, 0.3])
            next_table[-1, ind] = num
            cls.move_map_element(MODE, move_map, ind, MODE, MODE - 1, action)

            if action == 'top':
                state.loc[MODE, ind] = num
            elif action == 'left':
                state.loc[MODE - 1 - ind, MODE] = num
            elif action == 'bottom':
                state.loc[-1, MODE - 1 - ind] = num
            elif action == 'right':
                state.loc[ind, -1] = num
            else:
                raise ValueError('"action" should in ["top", "left", "bottom", "right"]')

        next_state = pd.DataFrame(
            np.zeros((MODE+2, MODE+2), dtype=np.int64), columns=list(range(-1, MODE+1)),
            index=list(range(-1, MODE+1)))
        next_state.iloc[1:MODE + 1, 1:MODE + 1] = next_table

        next_state = cls.state_set(MODE, action, next_state, next_table)

        return state, next_state, move_map

    def build_background(self, canvas):
        for i in range(self.MODE):
            for j in range(self.MODE):
                rect = (self.edge + np.array([self.grid[i], self.grid[j]])).reshape(-1)
                canvas.create_polygon(*rect, fill='DarkGray')
        self.refresh_canvas(canvas)

    def refresh_canvas(self, canvas):
        for value in self.collection.values():
            canvas.delete(value[0])
            canvas.delete(value[1])
        self.collection = {}
        for i in range(-1, self.MODE+1):
            for j in range(-1, self.MODE+1):
                if self.state.loc[i, j] != 0:
                    rect = (self.edge + np.array([self.grid[j], self.grid[i]])).reshape(-1)
                    element = canvas.create_polygon(
                        *rect, fill=color_bar[self.state.loc[i, j]], tag='{},{}'.format(i, j))
                    text = canvas.create_text(self.grid[j], self.grid[i], text=str(self.state.loc[i, j]),
                                              fill='White', tag='{}.{}'.format(i, j))
                    self.collection[(i, j)] = element, text

    def update_panel(self, canvas, label, action):
        state, next_state, move_map = self.predict_state(self.MODE, action, self.state)
        self.state = state
        self.refresh_canvas(canvas)

        move_message = {}
        for key, value in move_map.items():
            d_i = (self.grid[value[0]] - self.grid[key[0]]) / self.FRAMES
            d_j = (self.grid[value[1]] - self.grid[key[1]]) / self.FRAMES
            move_message['{},{}'.format(key[0], key[1])] = (d_i, d_j)

        for i in range(self.FRAMES):
            for key, value in move_message.items():
                canvas.move(key, value[1], value[0])
                canvas.move(key.replace(',', '.'), value[1], value[0])
            canvas.update()
            if i != self.FRAMES - 1:
                time.sleep(self.PAUSE)

        self.state = next_state
        self.refresh_canvas(canvas)

        score = np.sum((self.state.values[1:-1, 1:-1] // 4) ** 2)
        label["text"] = 'Score: ' + str(score)

        with open(r'./SaveData.dat', 'wb') as f:
            pickle.dump(self.state, f)

    def refresh_panel(self, canvas, label):
        init_ids = np.random.choice(self.MODE, 4, replace=False)
        while init_ids[0] == init_ids[2] and init_ids[1] == init_ids[3]:
            init_ids = np.random.choice(self.MODE, 4, replace=False) + 1
        self.state = pd.DataFrame(
            np.zeros((self.MODE + 2, self.MODE + 2), dtype=np.int64),
            columns=list(range(-1, self.MODE + 1)),
            index=list(range(-1, self.MODE + 1))
        )
        self.state.loc[init_ids[0], init_ids[1]] = 2
        self.state.loc[init_ids[2], init_ids[3]] = 2

        self.refresh_canvas(canvas)
        label["text"] = 'Score: '

        with open(r'./SaveData.dat', 'wb') as f:
            pickle.dump(self.state, f)
