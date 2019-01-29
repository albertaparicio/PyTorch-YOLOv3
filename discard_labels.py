# Discard labels smaller than 15px
#
# Author: Albert Aparicio <albert.aparicio.-nd@disneyresearch.com>
import argparse
import imghdr
import os
import struct


def get_image_size(fname):
    """Determine the image type of fhandle and return its size.
    from draco

    Code from Fred the Fantastic @ StackOverflow
    https://stackoverflow.com/a/20380514
    """
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def main(args):
    valid_images = []
    valid_labels = []

    with open(args.image_file, 'r') as i_f, open(args.label_file, 'r') as l_f:
        for imname, labname in zip(i_f.readlines(), l_f.readlines()):
            imname = imname.strip()
            labname = labname.strip()

            # Read image size
            try:
                w_im, h_im = get_image_size(imname)
            except TypeError:
                continue

            # Read label data
            valid_lines = []

            with open(labname, 'r') as lab:
                for lab_line in lab.readlines():
                    label = lab_line.strip().split(' ')

                    w_px = float(label[3]) * w_im
                    h_px = float(label[4]) * h_im

                    if not (w_px < 15 or h_px < 15):
                        valid_lines.append(label)
                    # else:
                    #     print(w_px)
                    #     print(h_px)

            if len(valid_lines) > 0:
                # Save image name to valid_images
                valid_images.append(imname)

                # Save valid lines to file
                fname, _ = os.path.splitext(labname)

                with open(fname + '.clean.txt', 'w') as f:
                    for line in valid_lines:
                        f.write(' '.join(line) + '\n')

                # Save file name of valid lines in valid_labels
                valid_labels.append(fname + '.clean.txt')

    assert len(valid_images) == len(valid_labels)

    imfile, _ = os.path.splitext(args.image_file)
    labfile, _ = os.path.splitext(args.label_file)

    with open(imfile + '.clean.txt', 'w') as f:
        for line in valid_images:
            f.write(line + '\n')

    with open(labfile + '.clean.txt', 'w') as f:
        for line in valid_labels:
            f.write(line + '\n')


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
