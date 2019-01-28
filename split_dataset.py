import argparse
import os
from random import shuffle

import numpy as np


def main(args):
    image_basename, _ = os.path.splitext(args.image_file)
    label_basename, _ = os.path.splitext(args.label_file)

    # Read file lists
    with open(args.image_file, 'r') as f:
        images_list = {}

        for image_file in f.readlines():
            fname, _ = os.path.splitext(os.path.basename(image_file.strip()))

            images_list[fname] = image_file.strip()

    with open(args.label_file, 'r') as f:
        # Read label contents
        label_files = {}
        labels_list = {}
        for label_file in f.readlines():
            fname, _ = os.path.splitext(os.path.basename(label_file.strip()))

            label_files[fname] = label_file.strip()

            with open(label_file.strip(), 'r') as l_f:
                label_content = l_f.readline()

                if len(label_content) > 0:
                    label_id = int(label_content.split()[0])

                    if label_id not in labels_list.keys():
                        labels_list[label_id] = []

                    labels_list[label_id].append(fname)

    train_res = []
    valid_res = []
    test_res = []

    for l_id, l_files in labels_list.items():
        idx = list(range(len(l_files)))
        shuffle(idx)

        train_num = int(round(len(l_files) * args.train_split, 0))
        valid_num = int(round(len(l_files) * args.valid_split, 0))
        test_num = len(l_files) - train_num - valid_num

        train_idx = idx[:train_num]
        valid_idx = idx[train_num:train_num + valid_num]
        test_idx = idx[-test_num:]

        train_res.extend(np.array(l_files)[train_idx].tolist())
        valid_res.extend(np.array(l_files)[valid_idx].tolist())
        test_res.extend(np.array(l_files)[test_idx].tolist())

    with open(image_basename + '.train.txt', 'w'
              ) as i, open(label_basename + '.train.txt', 'w') as l_f:
        for f in train_res:
            i.write(images_list[f] + '\n')
            l_f.write(label_files[f] + '\n')

    with open(image_basename + '.valid.txt', 'w'
              ) as i, open(label_basename + '.valid.txt', 'w') as l_f:
        for f in valid_res:
            i.write(images_list[f] + '\n')
            l_f.write(label_files[f] + '\n')

    with open(image_basename + '.test.txt', 'w'
              ) as i, open(label_basename + '.test.txt', 'w') as l_f:
        for f in test_res:
            i.write(images_list[f] + '\n')
            l_f.write(label_files[f] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split data into training, validation and test partitions. '
                    'The splitting percentage is maintained within each class.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--image-file', type=str, required=True,
                        help='File with absolute filenames of dataset images')
    parser.add_argument('-l', '--label-file', type=str, required=True,
                        help='File with absolute filenames of dataset labels')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Fraction of images to be put into training set')
    parser.add_argument('--valid-split', type=float, default=0.2,
                        help='Fraction of images to be put into validation set')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of images to be put into test set')

    opt = parser.parse_args()

    main(opt)

exit()
