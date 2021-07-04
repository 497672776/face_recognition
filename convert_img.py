# coding=utf-8
import numpy as np
import os
import cv2


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


def convert_img(path, new_img_path):
    file_list = os.listdir(path)
    data_np = []
    for file in file_list:
        input_file_name = path
        input_file_name = input_file_name + '/' + file
        nx = 128
        ny = 128
        file_load = np.fromfile(input_file_name, dtype='u1')
        if file_load.shape[0] != nx * ny:
            file_load = np.reshape(file_load, (512, 512))
            file_load = cv2.resize(file_load, (128, 128))
        file_load = np.reshape(file_load, (nx, ny, 1))
        cv2.imwrite(new_img_path + str(file) + '.jpg', file_load)
        data_np.append(file_load)
    data_np = np.array(data_np)
    return data_np


new_img_path = './img/'
data_path = 'F:/Course/PR/input/face_recognition/face/rawdata'
mkdir(new_img_path)
data_set = convert_img(data_path, new_img_path)
