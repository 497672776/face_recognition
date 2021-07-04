# coding=utf-8
import numpy as np
import os
import cv2
import re


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


def convert_img(raw_path, img_path):
    nx = 128
    ny = 128
    file_load = np.fromfile(raw_path, dtype='u1')
    if file_load.shape[0] != nx * ny:
        file_load = np.reshape(file_load, (512, 512))
        file_load = cv2.resize(file_load, (128, 128))
    file_load = np.reshape(file_load, (nx, ny, 1))
    cv2.imwrite(img_path, file_load)


def convert_img_label(feature_path, new_img_dir, raw_data_path):
    with open(feature_path) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            label_index = re.sub(r'\D', "", each_line)
            img_save_path = new_img_dir + sex + '/' + label_index + '.jpg'
            convert_img(raw_data_path + '/' + label_index, img_save_path)
        f.close()


classes = ['male', 'female']
classes_dict = {'male': 0, 'female': 1}

for item in classes:
    dir_path = './img2/'+item+'/'
    mkdir(dir_path)

new_img_dir = './img2/'
raw_data_path = 'F:/Course/PR/input/face_recognition/face/rawdata'
file_name_DR = 'F:/Course/PR/input/face_recognition/face/faceDR'
file_name_DS = 'F:/Course/PR/input/face_recognition/face/faceDS'
convert_img_label(file_name_DR, new_img_dir, raw_data_path)
