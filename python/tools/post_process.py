#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import ast
import re

# Changes the order of channels in an input image. Taken from original Fathom project
#
# command:
#   python3 post_process.py --file <path to results file> --shape n,c,h,w --zmajor

def convert_image(file_path, shape, datatype=np.uint8, zmajor=True):

    new_shape = [int(shape[0]),
                 int(shape[1]),
                 int(shape[2]),
                 int(shape[3])]

    arr = np.fromfile(file_path, dtype=datatype)
    data = np.reshape(arr, (new_shape[2], new_shape[3], new_shape[1]))
    if (zmajor):
        data = data.transpose([2, 0, 1])
    else:
        data = data.transpose([0, 1, 2])

    fp = open("output_transposed.dat", "wb")
    fp.write ((data.flatten()).astype(datatype).data)
    fp.close


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def main():
    parser = argparse.ArgumentParser(description='Convert the image used to a format suitable for KMB.')
    parser.add_argument('--file', type=str, required=True, help='an image to test the model')
    parser.add_argument('--shape', type=str)
    parser.add_argument('--zmajor', action='store_true')

    args = parser.parse_args()

    image_shape = args.shape.split(',')

    convert_image(args.file, image_shape, np.uint8)
    
if __name__ == "__main__":
    main()
