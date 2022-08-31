import os
import keras
import tkinter.ttk
import numpy as np
import scipy.io.wavfile
from tkinter import *

# real_dataset下的文件夹作为调制类型
classes = os.listdir('../Dataset/Real_dataset/')


# 预测函数
def predict():
    # 从输入的地方读取模型和代预测的.wav文件
    filepath = input_filepath.get()
    weights = input_weight.get()
    # 加载训练好的模型
    model = keras.models.load_model(weights)

    rate, data_raw = scipy.io.wavfile.read(filepath, mmap=False)
    data = np.expand_dims(data_raw, axis=0)
    class_guess_list = []
    class_guess_dict = {}

    # 进度条
    progressbarOne = tkinter.ttk.Progressbar(root)
    progressbarOne.place(relx=0.25, rely=0.65, relheight=0.05, relwidth=0.5)
    progressbarOne['maximum'] = 100
    progressbarOne['value'] = 0

    # 均分一百份，每份中读1024个点来分别进行预测，100个预测结果就是调制方式的百分比概率
    for n in range(1, 101):
        slice = int(
            np.random.choice(range((n - 1) * int((data.shape[1]) / 100), n * int((data.shape[1]) / 100)), size=1,
                             replace=False))
        if (slice + 1024) >= data.shape[1]:
            slice = data.shape[1] - 1025
        data_slice_test = data[:, slice:(slice + 1024), :]

        # 缩小样本信号幅度，使之大小和训练集数据相近
        data_final = shrink(data_slice_test)

        # 预测
        y_real_guess = model.predict(data_final)
        class_guess_list.append(classes[int(np.argmax(y_real_guess[0, :]))])

        progressbarOne['value'] += 1
        root.update()

    # 计算每种调制方式出现了多少次，也就是它的百分比概率
    for key in class_guess_list:
        class_guess_dict[key] = class_guess_dict.get(key, 0) + 1

    # 输出结果
    output_print = [f'     有 {value}% 的可能性是 {key}\n' for key, value in class_guess_dict.items()]
    output_print.insert(0, '该信号调制方式: \n')

    for i in range(len(output_print)):
        txt.insert(END, output_print[i])


# 幅度缩小函数
def shrink(data_big):
    data_temp = data_big.T
    data_small = np.zeros((2, 1024, 1))
    for i in range(data_temp.shape[0]):
        for j in range(data_temp.shape[1]):
            data_small[i][j] = float(data_temp[i][j] / 1000)
    return data_small.T


root = Tk()
root.title('调制识别预测')
root.geometry('720x480')  # 这里的乘号不是 * ，而是小写英文字母 x

# 标题
title = Label(root, text='''基于深度学习的调制识别项目''', font=48)
title.place(relx=0.30, rely=0.01, relheight=0.1, relwidth=0.4)

# 输入 （拖入文件功能暂未实现）
label_weight = Label(root, text='请输入模型路径 或 拖入文件 (.h5)', font=8)
label_weight.place(relx=0.1, rely=0.1, relwidth=0.5, relheight=0.1)
label_path = Label(root, text='请输入待预测文件路径 或 拖入文件 (.wav)', font=8)
label_path.place(relx=0.1, rely=0.3, relwidth=0.6, relheight=0.1)
input_weight = Entry(root)
input_weight.place(relx=0.12, rely=0.2, relwidth=0.76, relheight=0.08)
input_filepath = Entry(root)
input_filepath.place(relx=0.12, rely=0.4, relwidth=0.76, relheight=0.08)

# 检测按钮，启动检查
btn = Button(root, text='检测', command=predict, font=48)
btn.place(relx=0.38, rely=0.52, relwidth=0.2, relheight=0.1)

# 输出
txt = Text(root)
txt.place(relx=0.12, rely=0.75, relheight=0.2, relwidth=0.76)

root.mainloop()
