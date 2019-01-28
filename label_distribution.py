# Analyze distribution of images per class
#
# Author: Albert Aparicio <albert.aparicio.-nd@disneyresearch.com>

import argparse
import json
import operator
from collections import OrderedDict


def main(args):
    class_labels = {}
    class_numbers = {}
    with open(args.image_file, 'r') as i_f, open(args.label_file, 'r') as l_f:
        for imname, labname in zip(i_f.readlines(), l_f.readlines()):
            imname = imname.strip()
            labname = labname.strip()

            with open(labname, 'r') as lab:
                lab_line = list(lab.readlines())[0]
                label = lab_line.strip().split(' ')

                class_num = int(label[0])

                if class_num not in class_labels.keys():
                    class_labels[class_num] = []
                    class_numbers[class_num] = 0

                class_numbers[class_num] += 1
                class_labels[class_num].append({
                    'img': imname,
                    'txt': labname
                })

    print(f'Max: {max([v for v in class_numbers.values()])}')
    print(f'Min: {min([v for v in class_numbers.values()])}')

    sorted_d = OrderedDict(sorted(class_numbers.items(), key=operator.itemgetter(1), reverse=True))

    with open('data/class_images.json', 'w') as j:
        json.dump(sorted_d, j, indent=4)

    with open('data/class_distribution.json', 'w') as j:
        json.dump(class_labels, j, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discard labels smaller than 15px',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--image-file', type=str, required=True,
                        help='File with absolute filenames of dataset images')
    parser.add_argument('-l', '--label-file', type=str, required=True,
                        help='File with absolute filenames of dataset labels')
    parser.add_argument('--min-size', type=int, default=15,
                        help='Minimum size for a label not to be discarded')

    opt = parser.parse_args()

    main(opt)

exit()
