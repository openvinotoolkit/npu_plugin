import os
import sys
import glob
import getopt

def printHelp() :
    print("** Run inference using InferenceManagerDemo and movisim emulator **\n"
        " 1. Pre-requirements:\n"
        "        Python3 should be installed\n"
        "        Your environment should provide following variables:\n"
        "        - WORKSPACE (path to dir, that contains vpuip_2)\n"
        "        - MV_TOOLS_DIR (path to dir, that contains MV_TOOLS)\n"
        "        - MV_TOOLS_VERSION\n"
        "    * Using script:\n"
        "        - `python3 run_MoviSim.py -n<path_to_vpuip_blob_compiled for_3720_arch> -i<path_to_input_0_blob> -i<path_to_input_1_blob> ... -o<path_where_output_0_blob_will_be_stored> -o<path_where_output_1_blob_will_be_stored> ... `\n"
        "        - Examples:\n"
        "        python3 run_MoviSim.py -n/home/Nets-Validation/MTL-NetTest-Validate/./MTL_por_caffe2_FP16-INT8_resnet-18-pytorch_MLIR.blob -i_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_input_0_case_0.blob -o_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_movisim_output_0_case_0.blob\n"
        "        - Result:\n"
        "        `_MTL_por_caffe2_FP16_INT8_resnet_18_pytorch_MLIR_movisim_output_0_case_0.blob` file will be created")


pathToVpuip2HomeDir = os.getenv('WORKSPACE', 'workspace_does_not_set') + '/vpuip_2'
pathToMoviSimDir = os.getenv('MV_TOOLS_DIR', 'mv_tools_dir_does_not_set') + '/' + os.getenv('MV_TOOLS_VERSION', 'mv_tools_version_does_not_set') + '/linux64/bin'
pathToNetworkBlob = ''
pathToInputBlobs = []
pathToOutputBlobs = []

print("Run_MoviSim.py srcipt has been called")
print("Path to vpuip_2 home dir:", pathToVpuip2HomeDir)
print("Path to moviTools dir", pathToMoviSimDir)

pathToIEDemoDir = pathToVpuip2HomeDir + "/application/demo/InferenceManagerDemo"

fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]

unixOptions = "hv:m:n:i:o:"
gnuOptions = ["help", "path_to_moviSim_dir", "path_to_vpuip_2_home_dir", "path_to_network_blob", "path_to_input_blobs", "path_to_output_blobs"]

try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    print (str(err))
    sys.exit(2)


for currentArgument, currentValue in arguments:
    if currentArgument in ("-h", "--help"):
        printHelp()
        exit()
    elif currentArgument in ("-m", "--path_to_moviSim_dir"):
        print ("path_to_moviSim_dir (%s)" % currentValue)
        pathToMoviSimDir = currentValue
    elif currentArgument in ("-v", "--path_to_vpuip_2_home_dir"):
        print ("path_to_vpuip_2_home_dir (%s)" % currentValue)
        pathToVpuip2HomeDir = currentValue
    elif currentArgument in ("-n", "--path_to_network_blob"):
        print ("path_to_network_blob (%s)" % currentValue)
        pathToNetworkBlob = currentValue
    elif currentArgument in ("-i", "--path_to_input_blob"):
        print ("path_to_input_blobs (%s)" % currentValue)
        pathToInputBlobs.append(currentValue)
    elif currentArgument in ("-o", "--path_to_output_blob"):
        print ("path_to_output_blobs (%s)" % currentValue)
        pathToOutputBlobs.append(currentValue)

# copy network blob
command = "cp " + pathToNetworkBlob + " " + pathToIEDemoDir + "/test.blob"
print(command)
os.system(command)

# remove all old input/output file
removeList = glob.glob(pathToIEDemoDir + '/input-*.bin')
removeList.extend(glob.glob(pathToIEDemoDir + '/output-*.bin'))

for filePath in removeList:
    try:
        print("Removing file : ", filePath)
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# copy inputs
inputsCounter = 0
for i in pathToInputBlobs :
    command = "cp " + i + " " + pathToIEDemoDir + "/input-" + str(inputsCounter) + ".bin"
    print(command)
    os.system(command)
    inputsCounter += 1

# prepare config
command = "cd " + pathToIEDemoDir + " && make -f Makefile prepare-kconfig"
print(command)
os.system(command)

# make elf
command = "cd " + pathToIEDemoDir + " && make -j8 CONFIG_FILE=.config_sim_3720xx "
print(command)
os.system(command)

# run inference
command = "cd " + pathToIEDemoDir + " && " + pathToMoviSimDir + "/moviSim -cv:3700xx -nodasm -q -l:LRT:./mvbuild/3720/InferenceManagerDemo.elf"
print(command)
os.system(command)

# copy inference result
outputFilesList = glob.glob(pathToIEDemoDir + '/output-*.bin')

for outputFile, resultPath in zip(outputFilesList, pathToOutputBlobs):
    try:
        print("output files:" + outputFile)
        command = "cp " + outputFile + " " + resultPath
        print(command)
        os.system(command)
    except:
        print("Can't copy output blob to result path : ", outputFile)
        
if(len(outputFilesList) != len(pathToOutputBlobs)) :
        print("Error: number of outputs <{0}> doesn't match with provided path_to_output_blobs <{1}>".format(len(outputFilesList), len(pathToOutputBlobs)));

