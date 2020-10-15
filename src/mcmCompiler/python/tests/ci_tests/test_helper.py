import sys, os
import subprocess
import csv
import re


# Some external defines
MAX_UINT32 = 4294967295
RESNET_CMX = 4*918
CONG_CMX = 4096** 1024

# Error Code Guide
RESULT_SUCCESS  = 0
FAIL_GENERAL    = 1
FAIL_RESERVED   = 2
FAIL_COMPILER   = 3  # Compiler fails to create graphfile or emulator files
FAIL_RUNTIME    = 4  # Runtime fails to generate results file (NCE2Task_network_out.bin, output.dat)
FAIL_VALIDATION = 5  # Correctness Error

# Internal Defines
MDK_ROOT = os.environ["MDK_HOME"]
NZOO_ROOT = os.environ["NZOO_ROOT"]
POC_ROOT = MDK_ROOT+'/projects/Fathom/'

# which runtime do you want to use? (PoC/Production)
#APP_ROOT = MDK_ROOT + "/testApps/kmb/tapeout_so/tc_kmb_ma2490/ts_nce_bm_blob/"  # PoC Runtime
APP_ROOT = MDK_ROOT + "/testApps/components/NeuralNet/008_demo_ncelib_LNN/conv_3l_16wi/" # Prod Runtime

PWD = os.path.dirname(os.path.realpath(__file__)) + "/"
CWD = os.getcwd() + "/"


header = lambda x : "========= " + x + " ========="

def skip():
    if os.getenv('RUN_FAILED', '0') == '0' or os.getenv('RUN_FAILED', '0') == '':
        sys.exit(123)

def generate_model(csv_file, network=False, simplify=False, overwrite=True):
    print(header("Generate Model"))
    sys.path.insert(0, NZOO_ROOT + '/tools/kmb_test_generation')
    from run_testfile import main as genModel

    class arg():
        def __init__(self, f, network, simplify=True):
            self.input_file=os.path.abspath(f)
            self.network=network
            self.simplify=simplify
            self.output_name="graph"

    if (".py" in csv_file):
        # This isn't a Csv file!
        print("Python")
        cmd = ["python3" , csv_file]
        print(cmd)
        code = subprocess.check_output(cmd)
        print("\n\nPath:", code)

        code = str(code.decode('ascii'))
        print("not b",code)
        m = re.search("FILE:(.+)", code)
        if m:
            found = m.group(1)
            print("found:", found)
            return found
        else:
            print("Did not detect string with prefix 'FILE' from .py")
            sys.exit(7) # Unsupported

    a = arg(csv_file, network, simplify=simplify)
    if overwrite:
        model = genModel(a)
    else:
        # re-use the existing model from previous run (due to random weights)
        model = generate_model_path(a)

    print ("Model: ", model)
    return model

def checkExists(file, errorcode=1, aliases=[]):
    # check for array of files and return first found
    aliases.append(file)
    for x in aliases:
        if os.path.isfile(x):
            return x

    print("File Not Produced", aliases)
    sys.exit(errorcode)

def compile_graphFile(model, nClusters, nDPU, strategyFile, cmx=4*918, emulator=True, cpp=False):
    if emulator:
        extra = " with Emulation"
    else:
        extra = ""
    print(header("Compile" + extra))

    sys.path.insert(0, POC_ROOT+"src2/")
    from Fathom import main as genGraph
    from Models.EnumDeclarations import OperationMode, Parser
    import Controllers.Globals as GLOBALS

    GLOBALS.USING_KMB = True
    GLOBALS.INPUT_IN_INTERLEAVED = False
    GLOBALS.OPT_SCHEDULER = True
    GLOBALS.n_DPE = 256
    GLOBALS.CMX_SIZE = 512 * 1024

    class arg():
        def __init__(self, f, cluster, dpu, sf, cmx, emulator=True, cpp=False):

            ddr = 4*1024
            set_layout = [
                (0, 1, 2, 3),
                (0, 2, 1, 3),
                (0, 2, 3, 1),
                (0, 3, 2, 1),
                (0, 1, 3, 2)
            ]

            # Customized
            self.net_description=f
            self.nClusters = cluster
            self.nDPU = dpu
            self.strategy_file = sf
            self.sparsity=False
            self.cmx=cmx * 1024
            self.ddr_heap=ddr*1024*1024
            self.ddr_bss=ddr*1024*1024
            self.strategy=0 #"Clustering"
            self.validation_type=3
            self.emulator=emulator
            self.barrier_reutilization=False
            self.scheduler=None
            self.raw_scale=1
            self.prefetching = 2
            # Defaults
            self.nWorkloads=[]
            self.save_input="InputTensor.bin"
            self.save_output="OutputTensor.bin"
            self.save_weights="weights.caffemodel"
            self.cpp=cpp
            self.comp_descriptor=os.environ["MCM_HOME"] + "/config/compilation/debug_ma2490.json"
            self.mcm_loglevel=None
            self.no_csv=False
            self.verbose=False
            self.class_test_threshold=0.2
            self.kmb=True
            self.blob_name="vpu2.blob"
            self.outputs_name=None
            self.outputs_location=CWD+"output/"
            self.channel_swap=None
            self.mean=None
            self.net_weights=None
            self.mode = OperationMode.generation
            self.number_of_shaves=1
            self.parser = Parser.TensorFlowLite
            self.save_input="Fathom_expected.npy"
            self.image=None
            self.input_node_name=None
            self.output_node_name=None
            self.seed= -1
            self.accuracy_table = {'ALL': 1.0}
            self.input_layout = set_layout
            self.output_layout = set_layout
            self.expected_index=None
            self.fpga=None
            self.hw_perm=False
            self.input_size=None


    a = arg(model, nClusters, nDPU, strategyFile, cmx, emulator=emulator, cpp=cpp)

    if False:
        from pprint import pprint
        pprint(vars(a))
        quit()

    out_folder = CWD+"/output/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    genGraph(a)
    genGraph = "out"

    graphFile = out_folder+"vpu2.blob"   # Ideally returned from Fathom
    sim_result = out_folder+"expected_result_sim.dat"   # Ideally returned from Fathom
    # sim_result = out_folder+"../Fathom_simulation.npy"
    fw_result = out_folder+"../Fathom_expected.npy"

    if emulator:
        checkExists(sim_result, FAIL_COMPILER)
    checkExists(graphFile, FAIL_COMPILER)

    return graphFile, sim_result, fw_result

def execute_network(gf):
    print(header("Execute Network"))
    checkExists(gf, FAIL_COMPILER)

    from shutil import copyfile

    # clean generated files
    generated_files = ["vpu2.blob", "input.dat", "NCE2Task_network_out.bin", "output.dat", "expected_result_sim.dat"]
    for file in generated_files:
        try:
            os.remove(APP_ROOT + file)
        except FileNotFoundError:
            pass

    # directories - rmdir only deletes empty directories, it won't delete recursively
    try:
        from shutil import rmtree
        rmtree(APP_ROOT + "output", ignore_errors=True)
    except Exception:
        pass

    print("Copy:", gf, "to", APP_ROOT + "vpu2.blob")
    copyfile(gf, APP_ROOT + "vpu2.blob")
    print("Copy:", CWD+"output/input.dat", "to", APP_ROOT + "input.dat")
    copyfile(CWD+"output/input.dat", APP_ROOT + "input.dat")
    print("Copy:", CWD + "output/expected_result_sim.dat", "to", APP_ROOT + "expected_result_sim.dat")
    copyfile(CWD + "output/expected_result_sim.dat", APP_ROOT + "expected_result_sim.dat")

    moviSimPort=os.getenv('MOVISIM_PORT','30001')
    mvToolsV=os.getenv('MV_TOOLS_VERSION','Latest_195458')


    sPort = " srvPort=" + moviSimPort
    sIP = ""
    fpga = os.getenv('FPGA', 'None')
    print("FPGA: \'", fpga,"\'", type(fpga))
    if fpga != 'None' and fpga is not None:
        sIP=' srvIP=iirfpga' + str(os.getenv('FPGA')) + ".ir.intel.com"
        sPort=""

    command = "make run -j MV_SOC_REV=ma2490 MV_TOOLS_VERSION="+mvToolsV+" GRAPH_BIN=vpu2.blob GRAPH_BIN_PATH=. "+sPort + sIP

    print(">>>>>>>>>>>>", command)
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT, cwd=APP_ROOT)

    if code.returncode != 0:
        print("Execution Failure")
        exit(code.returncode)

    result_file = APP_ROOT + "NCE2Task_network_out.bin"
    result_file_alt = [APP_ROOT + "NCE2Task0_out.bin", APP_ROOT + "output.dat"]  # Prod runtime uses "output.dat"

    result_file = checkExists(result_file, FAIL_RUNTIME, aliases=result_file_alt)

    return result_file

def validate_files(ref, test):

    checkExists(ref, FAIL_COMPILER)
    checkExists(test, FAIL_RUNTIME)

    result = True

    print(header("Validate Result"))

    command = "python3 "+POC_ROOT+"src2/Validate.py --reference "+ref+" --testdata "+ test + " --dtype u8"

    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT)

    if code.returncode != 0:
        result = False

    # dump first few lines of the results files
    for f in [ref, test]:
        print("Dump of ", f)
        with open(f, "rb") as bf:
            for i in range(0, 2):  # lines to dump
                print("{:02X} ".format(i * 32), " ".join("{:02X}".format(x) for x in bf.read(32)))  # number of bytes to dump/line

    import zlib
    def crc(fileName):
        prev = 0
        for eachLine in open(fileName,"rb"):
            prev = zlib.crc32(eachLine, prev)
        return "%X"%(prev & 0xFFFFFFFF)

    print("CRC EXPECTED: ", crc(ref))
    print("CRC RESULT: ", crc(test))

    if crc(ref) == crc(test):
        sys.exit(RESULT_SUCCESS)
    else:
        sys.exit(FAIL_VALIDATION)

    return result


def generate_model_path(args):
    # loads the csv file and creates the model path
    model_name = "/quantized_model.tflite"
    if (args.simplify):
        model_name = "/simplified_model.tflite"

    folder_name = args.output_name
    if not args.network:
        import csv
        with open(args.input_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                folder_name = row["Name"].strip()
                break

    model = NZOO_ROOT + "/internal/unit_tests/CompilerTestsKmb/layers/" + folder_name + model_name
    return model
