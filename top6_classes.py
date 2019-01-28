import argparse
import json
import operator
from collections import OrderedDict


def main(args):
    with open(args.images_file, 'r') as j:
        class_num = OrderedDict(
            sorted(json.load(j).items(),
                   key=operator.itemgetter(1), reverse=True))

    with open(args.dist_file, 'r') as j:
        class_distro = json.load(j)

    with open(f'data/yolo_images.top{args.top_class}.txt', 'w') as f_i, open(
            f'data/yolo_labels.top{args.top_class}.txt', 'w') as f_l:
        for cl_num in list(class_num.keys())[:args.top_class]:
            class_data = class_distro[cl_num]

            for data_item in class_data:
                f_i.write(data_item['img'] + '\n')
                f_l.write(data_item['txt'] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discard labels smaller than 15px',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--images-file', type=str, required=True,
                        help='File with the number of images in each class')
    parser.add_argument('-d', '--dist-file', type=str, required=True,
                        help='File with the filenames of each class images')
    parser.add_argument('-t', '--top-class', type=int, default=6,
                        help='Top N classes that will be selected for training')

    opt = parser.parse_args()

    main(opt)

exit()
