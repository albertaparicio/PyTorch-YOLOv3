# Analyze distribution of images per class
#
# Author: Albert Aparicio <albert.aparicio.-nd@disneyresearch.com>
import argparse
import json


def main(args):
    class_labels = {}

    with open(args.label_file, 'r') as l_f:
        for labname in l_f.readlines():
            labname = labname.strip()

            with open(labname, 'r') as lab:
                lab_line = list(lab.readlines())[0]
                label = lab_line.strip().split(' ')

                class_num = int(label[0])

                if class_num not in class_labels.keys():
                    class_labels[class_num] = {
                        'num': 0,
                        'files': []
                    }

                class_labels[class_num]['num'] += 1
                class_labels[class_num]['files'].append(labname)

    # print(class_labels)

    print(f'Max: {max([v["num"] for v in class_labels.values()])}')
    print(f'Min: {min([v["num"] for v in class_labels.values()])}')

    with open('data/class_distribution.json', 'w') as j:
        json.dump(class_labels, j, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discard labels smaller than 15px',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('-i', '--image-file', type=str, required=True,
    #                     help='File with absolute filenames of dataset images')
    parser.add_argument('-l', '--label-file', type=str, required=True,
                        help='File with absolute filenames of dataset labels')
    parser.add_argument('--min-size', type=int, default=15,
                        help='Minimum size for a label not to be discarded')

    opt = parser.parse_args()

    main(opt)

exit()
