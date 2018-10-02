#! /usr/bin/env python3

# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import os
import sys
import argparse
import numpy as np

base = os.environ.get('MDK_HOME')
sys.path.append(os.path.join(base, "projects/Fathom/src2/"))

from Controllers.DataTransforms import *
import matplotlib.pyplot as plt

if sys.version_info[0] != 3:
    sys.stdout.write("Attempting to run with a version of Python != 3.x\n")
    sys.exit(1)

from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *
from Views.Validate import *
from Controllers.Args import coords
import Controllers.Globals as GLOBALS
from Controllers.Scheduler import load_myriad_config, load_network
from Controllers.PingPong import ppInit

major_version = np.uint32(2)
release_number = np.uint32(0)


def run_blob_myriad(blob_path, image_path, inputTensorShape, outputTensorShape, arguments):
    """
    Runs our myriad elf
    :param elf: path to elf.
    :param blob: blob object.
    :return:

    Side Effects: Creates some .npy files containing versions of the myriad output before and after transformation.
    """

    global device
    global myriad_debug_size

    debug = True
    hw = False

    np.random.seed(19)

    f = open(blob_path, 'rb')
    blob_file = f.read()
    import binascii
    print("CRC blob_file: ", binascii.crc32(blob_file))

    if device is None:
        devices = mvncapi.enumerate_devices()
        if len(devices) == 0:
            throw_error(ErrorTable.USBError, 'No devices found')

        # Choose the first device unless manually specified
        if arguments.device_no is not None:
            device = mvncapi.Device(devices[arguments.device_no])
        else:
            device = mvncapi.Device(devices[0])

        try:
            device.open()
        except:
            throw_error(ErrorTable.USBError, 'Error opening device')
    #net.inputTensor = net.inputTensor.astype(dtype=np.float16)
    #input_image = net.inputTensor

    image_path = "Debug"

    if ".npy" in image_path:
        print("NUMPY")
        input_image = np.load("test.npy")
        input_data_layout = StorageOrder.orderZYX

    elif image_path is None or image_path == "Debug":
        input_image = np.random.uniform(0, 1, inputTensorShape).astype(np.float16)
        if (hw):
            # assume data in ZYX
            if (len(inputTensorShape) == 4):
                input_data = input_image.reshape(int(inputTensorShape[0]), int(inputTensorShape[3]), int(inputTensorShape[1]), int(inputTensorShape[2]))
            input_data_layout = StorageOrder.orderZYX
        else:
            # assume data in YXZ
            if (len(inputTensorShape) == 4):
                input_data = input_image.reshape(int(inputTensorShape[0]), int(inputTensorShape[1]), int(inputTensorShape[2]), int(inputTensorShape[3]))
            input_data_layout = StorageOrder.orderYXZ
        if debug:
            print("Input image shape", inputTensorShape)

        if (input_data_layout == StorageOrder.orderZYX):
            # transpose to YXZ
            if (len(inputTensorShape)==4):
                input_image = input_image.transpose([0, 2, 3, 1])
    else:
        input_image = parse_img(image_path,
                           [int(inputTensorShape[0]),
                            int(inputTensorShape[3]),
                            int(inputTensorShape[1]),
                            int(inputTensorShape[2])],
                           raw_scale=arguments.raw_scale,
                           mean=arguments.mean,
                           channel_swap=(2, 1, 0)).astype(np.float16)
        #input_image = plt.imread(image_path).astype(np.float16)
        #input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], input_image.shape[2]))
        input_image = input_image.transpose([0, 2, 3, 1])

    if GLOBALS.INPUT_IN_INTERLEAVED:
        # Convert to planar, pad, and then convert back to interleaved.
        s = input_image.shape

        # We reshape to the original dimensions to avoid errors
        # in different parts of the code.
        if len(s) == 4:
            # Recover interleaved shape and convert to planar
            si = (s[0], s[2], s[1], s[3])
            if si[2] == 3:
                input_image = input_image.reshape(si).transpose(0, 2, 1, 3)

        if len(s) == 3:
            # Recover interleaved shape and convert to planar
            si = (s[1], s[0], s[2])
            if si[1] == 3:
                input_image = input_image.reshape(si).transpose(1, 0, 2)

    if input_image.shape[1] == 3:
        padded_slice = np.zeros((input_image.shape[0],
                                1,
                                input_image.shape[2],
                                input_image.shape[3]),
                                dtype=float).astype(dtype=np.float16)
        input_image = np.append(input_image, padded_slice, axis=1)

        if GLOBALS.INPUT_IN_INTERLEAVED:
            s = input_image.shape
            # We reshape to the original dimensions to avoid errors
            # in different parts of the code.
            if len(s) == 4:
                input_image = input_image.transpose(0, 2, 1, 3).reshape(s)

            if len(s) == 3:
                input_image = input_image.transpose(1, 0, 2).reshape(s)

    #if arguments.save_input is not None:
    #    net.inputTensor.tofile(arguments.save_input)
    print("USB: Transferring Data...")
    if arguments.lower_temperature_limit != -1:
        device.set_option(
            mvncapi.DeviceOptionClass2.RW_TEMP_LIM_LOWER,
            arguments.lower_temperature_limit)
    if arguments.upper_temperature_limit != -1:
        device.set_option(
            mvncapi.DeviceOptionClass2.RW_TEMP_LIM_HIGHER,
            arguments.upper_temperature_limit)
    if arguments.backoff_time_normal != -1:
        device.set_option(
            mvncapi.DeviceOptionClass2.RW_BACKOFF_TIME_NORMAL,
            arguments.backoff_time_normal)
    if arguments.backoff_time_high != -1:
        device.set_option(
            mvncapi.DeviceOptionClass2.RW_BACKOFF_TIME_HIGH,
            arguments.backoff_time_high)
    if arguments.backoff_time_critical != -1:
        device.set_option(
            mvncapi.DeviceOptionClass2.RW_BACKOFF_TIME_CRITICAL,
            arguments.backoff_time_critical)
#    device.set_option(
#        mvncapi.DeviceOptionClass2.RW_TEMPERATURE_DEBUG,
#        1 if arguments.temperature_mode == 'Simple' else 0)
    graph = mvncapi.Graph("graph");
    graph.allocate(device, blob_file)

#    graph.set_option(
#        mvncapi.GraphOptionClass1.ITERATIONS,
#        arguments.number_of_iterations)
#    graph.set_option(
#        mvncapi.GraphOptionClass1.NETWORK_THROTTLE,
#        arguments.network_level_throttling)

    fifoIn = mvncapi.Fifo("fifoIn0", mvncapi.FifoType.HOST_WO)
    fifoOut = mvncapi.Fifo("fifoOut0", mvncapi.FifoType.HOST_RO)
    fifoIn.set_option(mvncapi.FifoOption.RW_DATA_TYPE, mvncapi.FifoDataType.FP16)
    fifoOut.set_option(mvncapi.FifoOption.RW_DATA_TYPE, mvncapi.FifoDataType.FP16)
    descIn = graph.get_option(mvncapi.GraphOption.RO_INPUT_TENSOR_DESCRIPTORS)
    descOut = graph.get_option(mvncapi.GraphOption.RO_OUTPUT_TENSOR_DESCRIPTORS)
    fifoIn.allocate(device, descIn[0], 2)
    fifoOut.allocate(device, descOut[0], 2)

    # input_image.fill(1)
    import binascii
    print("CRC IMAGE: ", binascii.crc32(input_image))


    for y in range(arguments.stress_full_run):
        if arguments.timer:
            import time
            ts = time.time()
        graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, input_image, None)
        try:
            myriad_output, userobj = fifoOut.read_elem()
        except Exception as e:
            print("GetResult exception")
            if e.args[0] == mvncapi.Status.MYRIAD_ERROR:
                debugmsg = graph.get_option(mvnc.DeviceOption.RO_DEBUG_INFO)
                throw_error(ErrorTable.MyriadRuntimeIssue, debugmsg)
            else:
                throw_error(ErrorTable.MyriadRuntimeIssue, e.args[0])

        if arguments.timer:
            ts2 = time.time()
            print("\033[94mTime to Execute : ", str(
                round((ts2 - ts) * 1000, 2)), " ms\033[39m")

        print("USB: Myriad Execution Finished")

    timings = graph.get_option(mvncapi.GraphOption.RO_TIME_TAKEN)
    if arguments.mode in [OperationMode.temperature_profile]:
        tempBuffer = device.get_option(mvncapi.DeviceOption.RO_THERMAL_STATS)
    throttling = device.get_option(
        mvncapi.DeviceOption.RO_THERMAL_THROTTLING_LEVEL)
    if throttling == 1:
        print("*********** THERMAL THROTTLING INITIATED ***********")
    if throttling == 2:
        print("************************ WARNING ************************")
        print("*           THERMAL THROTTLING LEVEL 2 REACHED          *")
        print("*********************************************************")

    print("Myriad Output: \n", myriad_output);
    print("Myriad Output: \n", myriad_output.shape);
    print("Output: ", outputTensorShape)
    myriad_output = myriad_output.reshape(outputTensorShape)

    if arguments.mode in [OperationMode.temperature_profile]:
        net.temperature_buffer = tempBuffer

    if arguments.save_output is not None:
        myriad_output.tofile(arguments.save_output)
        np.save("Fathom_result.npy", myriad_output)

    print("USB: Myriad Connection Closing.")
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()
    print("USB: Myriad Connection Closed.")
    return timings, myriad_output

def parse_args():
    parser = argparse.ArgumentParser(description="Script that runs a blob file on the Movidius Neural Compute Stick\n")
    #parser.add_argument('network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('blob', type=str, help='Blob file')
    parser.add_argument('in_s', type=str, help='Input tensor shape, format (n,w,h,c)')
    parser.add_argument('out_s', type=str, help='Output tensor shape, format (w,h,c)')
    parser.add_argument('-res', type=str, default='output', help='Output file name')
    parser.add_argument('-i', dest='image', type=str, default='Debug', help='Image to process')
    parser.add_argument('-S', dest='scale', type=float, help='Scale the input by this amount, before mean')
    parser.add_argument('-M', dest='mean', type=str, help='Numpy file or constant to subtract from the image, after scaling')
    parser.add_argument('-id', dest='expectedid', type=int, help='Expected output id for validation')
    parser.add_argument('-cs', dest='channel_swap', type=coords, default=(2,1,0), help="default: 2,1,0 for RGB-to-BGR; no swap: 0,1,2", nargs='?')
    parser.add_argument('-dn', dest='device_no', metavar='', type=str, nargs='?', help="Experimental flag to run on a specified stick.")
    parser.add_argument('-ec', dest='explicit_concat', action='store_true', help='Force explicit concat')
    parser.add_argument('--accuracy_adjust', type=str, const="ALL:256", default="ALL:1", help='Scale the output by this amount', nargs='?')
    parser.add_argument('--ma2480', action="store_true", help="Dev flag")
    parser.add_argument('--scheduler', action="store", help="Dev flag")
    parser.add_argument('-of', dest='save_output', type=str, default=None,
            help='File name for the myriad result output in numpy format.')
    parser.add_argument('-rof', dest='save_ref_output', type=str, default=None,
            help='File name for the reference result in numpy format')
    parser.add_argument('-metric', dest = 'metric', type = str,
            default = "top5", help = "Metric to be used for validation.\
                    Options: top1, top5 or accuracy_metrics, ssd_pred_metric")
    args = parser.parse_args()
    return args


class Arguments:
    def __init__(self):
        self.raw_scale = 1
        self.mean = None
        self.channel_swap = None
        self.acm = 0
        self.timer = None
        self.number_of_iterations = 2
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 1
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'validation'
        self.outputs_name = 'output'
        self.save_input = None
        self.save_output = None
        self.device_no = None
        self.exp_id = None
        self.seed = -1
        self.accuracy_table = {}
        self.ma2480 = True
        if args.accuracy_adjust != "":
            pairs = args.accuracy_adjust.split(',')
            for pair in pairs:
                layer, value = pair.split(':')
                self.accuracy_table[layer] = float(value)


def check_net(blob_path, image_path, input_shape_str, output_shape_str, output_filename):
    file_init()

    default_params = Arguments()
    GLOBALS.USING_MA2480 = True
    GLOBALS.OPT_SCHEDULER = False

    file_gen = True

    input_shape_substrs = input_shape_str.split(',')
    output_shape_substrs = output_shape_str.split(',')
    inputTensorShape = (int(input_shape_substrs[0][1:]), int(input_shape_substrs[1]), int(input_shape_substrs[2]), int(input_shape_substrs[3][0:-1]))
    outputTensorShape = (int(output_shape_substrs[0][1:]), int(output_shape_substrs[1]), int(output_shape_substrs[2][0:-1]))

    print("inputTensorShape", inputTensorShape)

    timings, myriad_output = run_blob_myriad(blob_path, image_path, inputTensorShape, outputTensorShape, default_params)

    myriad_output = storage_order_convert(myriad_output, StorageOrder.orderYXZ, StorageOrder.orderYXZ)
    np.save(output_filename + ".npy", myriad_output)

    return 0


if __name__ == "__main__":
    setup_warnings()
    print("Blob deploy tool\n")
    args = parse_args()
    quit_code = check_net(args.blob, args.image, args.in_s, args.out_s, args.res)
    quit(quit_code)
