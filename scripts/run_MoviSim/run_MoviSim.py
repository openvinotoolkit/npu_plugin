import os
import sys
import time
import glob
import getopt
import subprocess

path_to_vpuip_2_home_dir = os.getenv('WORKSPACE', 'workspace_doesnt_set') + '/vpuip_2'
path_to_moviSim_dir = os.getenv('MV_TOOLS_DIR', 'mv_tools_dir_doesnt_set') + '/' + os.getenv('MV_TOOLS_VERSION', 'mv_tools_version_doesnt_set') + '/linux64/bin'
path_to_network_blob = ''
path_to_input_blobs = []
path_to_output_blobs = []

print("Run_MoviSim.py srcipt has been called")
print("Path to vpuip_2 home dir:", path_to_vpuip_2_home_dir)
print("Path to moviTools dir", path_to_moviSim_dir)

path_to_IEDemo_dir = path_to_vpuip_2_home_dir + "/application/demo/InferenceManagerDemo"

fullCmdArguments = sys.argv
argumentList = fullCmdArguments[1:]

unixOptions = "hv:m:n:i:o:"
gnuOptions = ["help", "path_to_moviSim_dir", "path_to_vpuip_2_home_dir", "path_to_network_blob", "path_to_input_blobs", "path_to_output_blobs"]

for arguments in argumentList :
    print("args: ", arguments)

try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)


for currentArgument, currentValue in arguments:
    if currentArgument in ("-h", "--help"):
        print ("print help")
    elif currentArgument in ("-m", "--path_to_moviSim_dir"):
        print ("path_to_moviSim_dir (%s)" % currentValue)
        path_to_moviSim_dir = currentValue
    elif currentArgument in ("-v", "--path_to_vpuip_2_home_dir"):
        print ("path_to_vpuip_2_home_dir (%s)" % currentValue)
        path_to_vpuip_2_home_dir = currentValue
    elif currentArgument in ("-n", "--path_to_network_blob"):
        print ("path_to_network_blob (%s)" % currentValue)
        path_to_network_blob = currentValue
    elif currentArgument in ("-i", "--path_to_input_blob"):
        print ("path_to_input_blobs (%s)" % currentValue)
        path_to_input_blobs.append(currentValue)
    elif currentArgument in ("-o", "--path_to_output_blob"):
        print ("path_to_output_blobs (%s)" % currentValue)
        path_to_output_blobs.append(currentValue)

# check that all args has been provided

# copy network blob
command = "cp " + path_to_network_blob + " " + path_to_IEDemo_dir + "/test.blob"
print(command)
os.system(command)

# remove all old input/output file
removeList = glob.glob(path_to_IEDemo_dir + '/input-*.bin')
removeList.extend(glob.glob(path_to_IEDemo_dir + '/output-*.bin'))

for filePath in removeList:
    try:
        print("Removing file : ", filePath)
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# copy inputs
inputs_counter = 0
for i in path_to_input_blobs :
    command = "cp " + i + " " + path_to_IEDemo_dir + "/input-" + str(inputs_counter) + ".bin"
    print(command)
    os.system(command)
    inputs_counter += 1

## prepare IEDemo

## prepare config
command = "cd " + path_to_IEDemo_dir + " && make -f Makefile prepare-kconfig"
print(command)
os.system(command)

## make elf
command = "cd " + path_to_IEDemo_dir + " && make -j8 CONFIG_FILE=.config_sim_3720xx CONFIG_NN_LOG_VERBOSITY_LRT_WARN=y CONFIG_NN_LOG_VERBOSITY_LRT_INFO=n CONFIG_NN_LOG_VERBOSITY_LNN_WARN=y CONFIG_NN_LOG_VERBOSITY_LNN_INFO=n CONFIG_NN_LOG_VERBOSITY_SNN_WARN=y CONFIG_NN_LOG_VERBOSITY_SNN_INFO=n CONFIG_PROFILING_MASK=\"0b00000000\""
print(command)
os.system(command)

## run inference
command = "cd " + path_to_IEDemo_dir + " && " + path_to_moviSim_dir + "/moviSim -cv:3700xx -nodasm -q -l:LRT:./mvbuild/3720/InferenceManagerDemo.elf"
print(command)
os.system(command)

# copy inference result
outputFilesList = glob.glob(path_to_IEDemo_dir + '/output-*.bin')

for outputFile, resultPath in zip(outputFilesList, path_to_output_blobs):
    try:
        print("output files:" + outputFile)
        command = "cp " + outputFile + " " + resultPath
        print(command)
        os.system(command)
    except:
        print("Can't copy output blob to result path : ", outputFile)
        
if(len(outputFilesList) != len(path_to_output_blobs)) :
        print("Error: number of outputs <{0}> doesn't match with provided path_to_output_blobs <{1}>".format(len(outputFilesList), len(path_to_output_blobs)));

