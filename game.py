# @Author  : Zheng Li
# @Time    : 2018/7/23
# @Version : 1.0
# @Document: RL2048.ipynb
import tkinter as tk
import data_utils
import rl_utils


WIDTH, HIGHT = 300, 400
SPACING = WIDTH / 45
MODE = 4
FRAMES = 10
PAUSE = 0.01


window = tk.Tk()
window.title('RL2048')

window.geometry('{}x{}'.format(WIDTH+10, HIGHT))
window.iconbitmap('./icon/2048.ico')

canvas = tk.Canvas(window, bg='Gray', width=WIDTH, height=WIDTH)
canvas.grid(row=0)

frame = tk.Frame(width=WIDTH, height=HIGHT-WIDTH)
frame.grid(row=1)

score_label = tk.Label(frame, text='Score:', bg='green', font=('Arial', 8), width=12, height=2)
score_label.grid(row=0, column=0)

data = data_utils.SaveData(WIDTH, MODE, SPACING, FRAMES, PAUSE)
data.build_background(canvas)

top_icon = tk.PhotoImage(file='./icon/top.gif')
top = tk.Button(
    frame, relief='flat', image=top_icon, command=lambda: data.update_panel(canvas, score_label, 'top'))
top.grid(row=0, column=1)

left_icon = tk.PhotoImage(file='./icon/left.gif')
left = tk.Button(
    frame, relief='flat', image=left_icon, command=lambda: data.update_panel(canvas, score_label, 'left'))
left.grid(row=0, column=2)

bottom_icon = tk.PhotoImage(file='./icon/bottom.gif')
bottom = tk.Button(
    frame, relief='flat', image=bottom_icon, command=lambda: data.update_panel(canvas, score_label, 'bottom'))
bottom.grid(row=0, column=3)

right_icon = tk.PhotoImage(file='./icon/right.gif')
right = tk.Button(
    frame, relief='flat', image=right_icon, command=lambda: data.update_panel(canvas, score_label, 'right'))
right.grid(row=0, column=4)

refresh_icon = tk.PhotoImage(file='./icon/refresh.gif')
refresh = tk.Button(
    frame, relief='flat', image=refresh_icon, command=lambda: data.refresh_panel(canvas, score_label))
refresh.grid(row=1, column=2)

auto_icon = tk.PhotoImage(file='./icon/auto.gif')
auto = tk.Button(
    frame, command=lambda: data.update_panel(canvas, score_label, rl_utils.choose_activate(data)),
    relief='flat', image=auto_icon)
auto.grid(row=1, column=3)

window.mainloop()
