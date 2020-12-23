"""
    date:       2020/12/22 8:14 下午
    written by: neonleexiang
"""
import os


def write_lines_to_file(txt_path, save_path):
    input_file = open(txt_path, 'r')
    lines_count = 1     # to set the txt file name
    for each_line in input_file.readlines():
        output_name = 'training-' + str(lines_count).zfill(6)       # create the file name
        output_file = open(os.path.join(save_path, output_name), 'w')
        output_file.write(each_line)
        output_file.close()
        lines_count += 1
    input_file.close()
    print('each line has write into individual file!')


if __name__ == '__main__':

    txt_file_PATH = 'datasets/train_src.txt'
    save_file_PATH = 'training_data'

    if not os.path.exists(save_file_PATH):
        os.mkdir(save_file_PATH)

    write_lines_to_file(txt_file_PATH, save_file_PATH)
