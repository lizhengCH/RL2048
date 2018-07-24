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
MODE = 8
FRAMES = 10
PAUSE = 0.01

window.geometry('{}x{}'.format(WIDTH+10, HIGHT))
window.iconbitmap('./icon/2048.ico')

canvas = tk.Canvas(window, bg='Gray', width=WIDTH, height=WIDTH)
canvas.grid(row=0)

grid, unit, merge = utils.build_canvas(WIDTH, MODE, SPACING, FRAMES, PAUSE, canvas)

frame = tk.Frame(width=WIDTH, height=HIGHT-WIDTH)
frame.grid(row=1)

label = tk.Label(frame, text='Score:', bg='green', font=('Arial', 12), width=12, height=2)
label.grid(row=0, column=0)

top_icon = tk.PhotoImage(file='./icon/top.gif')
top = tk.Button(frame, command=lambda: merge('top'), relief='flat', image=top_icon)
top.grid(row=0, column=1)

left_icon = tk.PhotoImage(file='./icon/left.gif')
left = tk.Button(frame, command=lambda: merge('left'), relief='flat', image=left_icon)
left.grid(row=0, column=2)

bottom_icon = tk.PhotoImage(file='./icon/bottom.gif')
bottom = tk.Button(frame, command=lambda: merge('bottom'), relief='flat', image=bottom_icon)
bottom.grid(row=0, column=3)

right_icon = tk.PhotoImage(file='./icon/right.gif')
right = tk.Button(frame, command=lambda: merge('right'), relief='flat', image=right_icon)
right.grid(row=0, column=4)

refresh_icon = tk.PhotoImage(file='./icon/refresh.gif')
refresh = tk.Button(frame, command=lambda: None, relief='flat', image=refresh_icon)
refresh.grid(row=1, column=2)

auto_icon = tk.PhotoImage(file='./icon/auto.gif')
auto = tk.Button(frame, command=lambda: None, relief='flat', image=auto_icon)
auto.grid(row=1, column=3)

window.mainloop()