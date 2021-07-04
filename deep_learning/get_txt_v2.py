import re
import os
import random


def get_txt(file_path_1, file_path_2, img_root, quantity_proportion):
    count_for_equal = {'male': 0, 'female': 0}

    # get file_number
    with open(file_path_1) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            sex = each_line[begin + 6:end]
            if begin == -1:
                continue
            count_for_equal[sex] = count_for_equal[sex] + 1
        f.close()

    with open(file_path_2) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            count_for_equal[sex] = count_for_equal[sex] + 1
        f.close()

    min_class_num = min(count_for_equal.values())
    # get number of data in train, val, test
    train_number = int(min_class_num * quantity_proportion[0])
    test_number = int(min_class_num * quantity_proportion[1])
    val_number = min_class_num - train_number - test_number

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

    train_count = {'male': 0, 'female': 0}
    test_count = {'male': 0, 'female': 0}
    val_count = {'male': 0, 'female': 0}
    num_for_debug = 0
    with open(file_path_1) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            label_index = re.sub(r'\D', "", each_line)
            if train_count[sex] < train_number:
                txt_list[0].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                train_count[sex] = train_count[sex] + 1
            elif test_count[sex] < test_number:
                txt_list[1].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                test_count[sex] = test_count[sex] + 1
            elif val_count[sex] < val_number:
                txt_list[2].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                val_count[sex] = val_count[sex] + 1
            else:
                num_for_debug = num_for_debug + 1
        f.close()

    with open(file_path_2) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            label_index = re.sub(r'\D', "", each_line)
            if train_count[sex] < train_number:
                txt_list[0].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                train_count[sex] = train_count[sex] + 1
            elif test_count[sex] < test_number:
                txt_list[1].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                test_count[sex] = test_count[sex] + 1
            elif val_count[sex] < val_number:
                txt_list[2].write(img_root + str(label_index) + ' ' + str(label_num_dict[sex]) + '\n')
                val_count[sex] = val_count[sex] + 1
            else:
                num_for_debug = num_for_debug + 1
        f.close()
    print(num_for_debug)


# train test val
quantity_proportion = [0.8, 0.1, 0.1]
file_name_DR = 'F:/Course/PR/input/face_recognition/face/faceDR'
file_name_DS = 'F:/Course/PR/input/face_recognition/face/faceDS'
img_dir_path = 'F:/Course/PR/input/face_recognition/face/rawdata/'
get_txt(file_name_DR, file_name_DS, img_dir_path, quantity_proportion)

