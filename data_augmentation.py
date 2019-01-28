import argparse
import json

import cv2
import numpy as np

from data_aug import (Sequence, RandomScale, RandomRotate,
                      RandomTranslate)


def parse_yolo_coordinates(rows, img):
    width_img = img.shape[1]
    height_img = img.shape[0]
    ret = []
    for r in rows:

        a = dict()
        r = r.split(" ")
        if len(r) == 1:
            continue
        x_center = float(r[1])
        y_center = float(r[2])
        width = float(r[3])
        height = float(r[4])
        label = float(r[0])
        ret.append([width_img * (x_center - width / 2.0), height_img * (y_center - height / 2.0),
                    width_img * (x_center + width / 2.0), height_img * (y_center + height / 2.0),
                    label])
    y = np.array([np.array(xi) for xi in ret])
    return y


def write_yolo_coordinates(bboxes, img, out_file):
    width_img = img.shape[1]
    height_img = img.shape[0]

    # bb: x1 y1 x2 y2 c
    if len(bboxes) == 0:
        return False

    with open(out_file, "w") as a:
        for bb in bboxes:
            width = (bb[2] - bb[0]) / float(width_img)
            height = (bb[3] - bb[1]) / float(height_img)

            center_x = (bb[0] + bb[2]) / (2.0 * width_img)
            center_y = (bb[1] + bb[3]) / (2.0 * height_img)

            a.write(f'{int(bb[4])} {center_x} {center_y} {width} {height}\n')

    return True


def main(args):
    min_images = args.min_images

    with open(args.class_file, 'r') as j:
        info = json.load(j)

    for category in info:
        if len(info[category]) > min_images:
            print("Skipping " + category)
            continue
        else:
            print("Processing " + category)
            num_images = len(info[category])
            curr_image = 0
            while num_images <= min_images:
                path_img = info[category][curr_image % len(info[category])]['img']
                path_txt = info[category][curr_image % len(info[category])]['txt']
                img = cv2.imread(path_img)
                bboxes = parse_yolo_coordinates(open(path_txt, "r").read().split("\n"), img)
                transforms = Sequence(
                    [RandomScale(0.2, diff=True), RandomRotate(20), RandomTranslate(0.02)])

                og_img = img
                og_boxes = bboxes

                # Keep trying to transform if there are errors
                # foo
                # There is some randomness in the transformation, so we cannot
                # know when or what there will be errors with the bboxes
                while True:
                    try:
                        img, bboxes = transforms(og_img, og_boxes)

                        break
                    except IndexError:
                        continue

                status = write_yolo_coordinates(
                    bboxes, img,
                    path_txt.replace(".txt", "_" + str(curr_image) + ".txt"))

                if status:
                    cv2.imwrite(path_img.replace(".jpg", "_" + str(curr_image) + ".jpg"), img)

                curr_image = curr_image + 1
                num_images = num_images + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Discard labels smaller than 15px',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-c', '--class-file', type=str, required=True,
        help="Path to JSON file with the following structure:"
             "dict( 'className1':[{img:img1,txt:txt1},{img:img2,txt:txt2},...],"
             "      'className2':[{img:img1,txt:txt1},{img:img2,txt:txt2},...],"
             "       ...)")
    parser.add_argument(
        '-n', '--min-images', type=int, default=50,
        help='Minimum number of images that we want to have per class')

    opt = parser.parse_args()

    main(opt)
