from ModelBuilder import save_model
import tensorflow as tf
import numpy as np
import argparse
from convolution import conv
from pooling import pool
from eltwise import eltwise
from activations import activation

def gen_resnet(args, name, input_range = (0, 255)):

    with tf.Graph().as_default():
        image_shape = [1, args.y, args.x, args.c]
        images = tf.placeholder("float", image_shape, name="input")

        # Res2a
        res2a_branch1 = conv(images, 64, 256, 1, 1, 1, 1, "SAME", True, quantize=args.quantize, sparsity=args.sparsity, name = "res2a_branch1")
        res2a_branch2a = conv(images, 64, 64, 1, 1, 1, 1, "SAME", True, quantize=args.quantize, sparsity=args.sparsity, name = "res2a_branch2a")
        res2a_branch2b = conv(res2a_branch2a, 64, 64, 3, 3, 1, 1, "SAME", True, quantize=args.quantize, sparsity=args.sparsity, name = "res2a_branch2b")
        res2a_branch2c = conv(res2a_branch2b, 64, 256, 1, 1, 1, 1, "SAME", True, quantize=args.quantize, sparsity=args.sparsity, name = "res2a_branch2c")
        res2a = eltwise(res2a_branch1, res2a_branch2c, True, quantize=args.quantize, name = "res2a")
        
        sess = tf.Session('')
        sess.run(tf.global_variables_initializer())

        save_model(name, images, res2a, sess, args.quantize, args.use_tf_light, disable=args.simplify)

def define_and_parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, type=int,
                        help = "Input Channels")
    parser.add_argument("-y", required=True, type=int,
                        help = "Height")
    parser.add_argument("-x", required=True, type=int,
                        help = "Width")
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help = "Sparsity")

    parser.add_argument("--quantize", action='store_true',
                        help = "Quantize network")

    parser.add_argument("--use-tf-light", default=True,
                        help = "Use TFLight")

    parser.add_argument("--simplify",  action='store_true',
                        help = "Disable parts of the inference for debug")

    return parser.parse_args()

def main():

    args = define_and_parse_args()
    gen_resnet(args, "resnet50")

if __name__=='__main__':
    main()
