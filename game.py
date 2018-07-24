# @Author  : Zheng Li
# @Time    : 2018/7/23
# @Version : 1.0
# @Document: Reinforcement-learning-with-tensorflow-master.ipynb

import time
import numpy as np
import tkinter as tk
import utils

window = tk.Tk()
window.title('2048')

WIDTH, HIGHT = 300, 400
SPACING = WIDTH / 45
MODE = 4
FRAMES = 10
PAUSE = 0.01

window.geometry('{}x{}'.format(WIDTH+10, HIGHT))

canvas = tk.Canvas(window, bg='Gray', width=WIDTH, height=WIDTH)
canvas.grid(row=0)

grid, unit, merge = utils.build_canvas(WIDTH, MODE, SPACING, FRAMES, PAUSE, canvas)

frame = tk.Frame(width=WIDTH, height=HIGHT-WIDTH)
frame.grid(row=1)
top = tk.Button(frame, text='top', command=lambda: merge('top')).grid(row=0, column=0)
left = tk.Button(frame, text='left', command=lambda: merge('left')).grid(row=0, column=1)
bottom = tk.Button(frame, text='bottom', command=lambda: merge('bottom')).grid(row=0, column=2)
right = tk.Button(frame, text='right', command=lambda: merge('right')).grid(row=0, column=3)

window.mainloop()