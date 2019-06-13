import os, sys
import numpy as np
import argparse
import subprocess
import json
from shutil import copyfile

# Instructions
# required environmental variables are:
#   $GRAPHFILE path to the graphfile repo
#   $MDK_HOME path to your mdk repo
# you must have Flatbuffers installed and the flatc binary in your path
# No validation currently in this version.
# Requires a valid FPGA booking


# Internal Defines
MDK_ROOT = os.environ["MDK_HOME"]
APP_ROOT = MDK_ROOT + "/testApps/components/NeuralNet/008_demo_ncelib_LNN/conv_3l_16wi/" # Prod Runtime
PWD = os.path.dirname(os.path.realpath(__file__)) + "/"
CWD = os.getcwd() + "/"

def generate_input(args):
    print("Generating input file...")
    blob_path = args.blob

    # convert blob to json to read the input layer shape
    subprocess.run(
        ["flatc", "-t", os.path.expandvars("$GRAPHFILE/src/schema/graphfile.fbs"), "--strict-json", "--", blob_path])

    json_blob = os.path.splitext(blob_path)[0] + ".json"
    with open(json_blob) as json_file:
        data = json.load(json_file)
        inputTensorShape = data["header"]["net_input"][0]["dimensions"]

    # create random input image
    print("inputTensorShape ", inputTensorShape)
    np.random.seed(19)
    input_image = np.random.uniform(0, 1, inputTensorShape).astype(np.int32)  # <-- input shape datatype
    np.save('input.npy', input_image)

    # flatten file and save as input.dat
    fp = open("{}/{}.dat".format(CWD, "input"), 'wb')
    fp.write((input_image.flatten()).astype(input_image.dtype).data)
    fp.close()

def execute_network(args):
    print("Executing Network...")

    # clean generated files
    generated_files = ["mcm.blob", "input.dat", "NCE2Task_network_out.bin", "output.dat", "expected_result_sim.dat"]
    for file in generated_files:
        try:
            os.remove(APP_ROOT + file)
        except FileNotFoundError:
            pass

    print("Copy:", args.blob, "to", APP_ROOT + args.blob)
    copyfile(args.blob, APP_ROOT + args.blob)
    print("Copy:", "input.dat", "to", APP_ROOT + "input.dat")
    copyfile("input.npy", APP_ROOT + "input.dat")
    # print("Copy:", "expected_result_sim.dat", "to", APP_ROOT + "expected_result_sim.dat")
    # copyfile("expected_result_sim.dat", APP_ROOT + "expected_result_sim.dat")

    moviSimPort = os.getenv('MOVISIM_PORT', '30001')
    mvToolsV = os.getenv('MV_TOOLS_VERSION', 'Latest_195458')

    sIP = ' srvIP=' + str(args.fpga)
    command = "make run -j MV_SOC_REV=ma2490 MV_TOOLS_VERSION=" + mvToolsV + " GRAPH_BIN=" + args.blob + " BLOB_NAME=" + args.blob + " GRAPH_BIN_PATH=. " + sIP

    print(">>>>>>>>>>>>", command)
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT, cwd=APP_ROOT)

    if code.returncode != 0:
        print("Execution Failure")
        exit(code.returncode)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="kmb_run_blob.py runs a blob on the FPGA, generating a results file\n")
    parser.add_argument('--blob', dest='blob', type=str, nargs='?', help='Blob file to load. Default is ./mcm.blob',
                        default='mcm.blob', const='mcm.blob')
    parser.add_argument('--fpga', dest='fpga', type=str, nargs='?', help='FPGA to run the blob on.')
    parser.add_argument('--version', action="store_true", help='Print version info.')
    args = parser.parse_args()

    if args.version:
        print("0.0.1")
        quit()
    if args.fpga is None:
        print('The FPGA arg is required eg, --fpga iirfpga017.ir.intel.com')
        sys.exit(2)

    generate_input(args)
    execute_network(args)