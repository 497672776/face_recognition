import re
import os
import random


def get_txt(file_path_1, file_path_2, img_root, quantity_proportion):
    # get file_number
    file_number = 0
    with open(file_path_1) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            file_number = file_number + 1
        f.close()

    with open(file_path_2) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            file_number = file_number + 1
        f.close()

    # get number of data in train, val, test
    train_number = int(file_number * quantity_proportion[0])
    test_number = int(file_number * quantity_proportion[1])
    val_number = file_number - train_number - test_number

    # 创建 txt 文件
    txt_names = ['./train.txt', './test.txt', './val.txt']
    for txt_name in txt_names:
        if os.path.isfile(txt_name):
            os.remove(txt_name)
    train_txt = open(txt_names[0], 'a')
    test_txt = open(txt_names[1], 'a')
    val_txt = open(txt_names[2], 'a')
    txt_list = [train_txt, test_txt, val_txt]
    label_num_dict = {'male': 0, 'female': 1}
    train_count = 0
    test_count = 0
    val_count = 0
    with open(file_path_1) as f:
        for each_line in f:
            rand = random.randint(0, 2)
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            label_index = re.sub(r'\D', "", each_line)
            if train_count < train_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                train_count = train_count + 1
            elif test_count < test_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                test_count = test_count + 1
            elif val_count < val_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                val_count = val_count + 1
            else:
                print("it is bug!")
        f.close()

    with open(file_path_2) as f:
        for each_line in f:
            rand = random.randint(0, 2)
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            label_index = re.sub(r'\D', "", each_line)
            if train_count < train_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                train_count = train_count + 1
            elif test_count < test_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                test_count = test_count + 1
            elif val_count < val_number:
                txt_list[rand].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                val_count = val_count + 1
            else:
                print("it is bug!")
        f.close()


# train test val
quantity_proportion = [0.8, 0.1, 0.1]
file_name_DR = r'F:\Course\PR\input\人脸图像识别\人脸图像识别\face\faceDR'
file_name_DS = r'F:\Course\PR\input\人脸图像识别\人脸图像识别\face\faceDS'
img_dir_path = 'F:/Course/PR/input/人脸图像识别/人脸图像识别/face/rawdata/rawdata/'
get_txt(file_name_DR, file_name_DS, img_dir_path, quantity_proportion)

