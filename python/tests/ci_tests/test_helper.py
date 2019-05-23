import sys, os
import subprocess
import csv


# Some external defines
MAX_UINT32 = 4294967295
RESNET_CMX = 4*918
CONG_CMX = 4096** 1024


# Internal Defines
MDK_ROOT = os.environ["MDK_HOME"]
NZOO_ROOT = os.environ["NZOO_ROOT"]
POC_ROOT = MDK_ROOT+'/projects/Fathom/'

# which runtime do you want to use? (PoC/Production)
#APP_ROOT = MDK_ROOT + "/testApps/keembay/tapeout_so/tc_keembay_ma2490/ts_nce_bm_blob/"  # PoC Runtime
APP_ROOT = MDK_ROOT + "/testApps/components/NeuralNet/008_demo_ncelib_LNN/mytest/" # Prod Runtime

PWD = os.path.dirname(os.path.realpath(__file__)) + "/"
CWD = os.getcwd() + "/"


header = lambda x : "========= " + x + " ========="

def skip():
    if os.getenv('RUN_FAILED', '0') == '0':
        sys.exit(123)

def generate_model(csv_file, network=False):   
    print(header("Generate Model")) 
    sys.path.insert(0, NZOO_ROOT + '/tools/kmb_test_generation')
    from run_testfile import main as genModel 

    class arg():
        def __init__(self, f, network):
            self.input_file=os.path.abspath(f)
            self.network=network
            self.simplify=False
            self.output_name="graph"

    a = arg(csv_file, network)
    model = genModel(a)
    
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
            self.ma2480=False
            self.cpp=cpp
            self.comp_descriptor=os.environ["MCM_HOME"] + "/config/compilation/debug_ma2490.json"
            self.mcm_loglevel=None
            self.no_csv=False
            self.verbose=False
            self.class_test_threshold=0.2
            self.kmb=True
            self.blob_name= "vpu2.blob"
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
    if (cpp==True):
        graphFile_mcm = out_folder+"blob.bin"
    sim_result = out_folder+"expected_result_sim.dat"   # Ideally returned from Fathom
    # sim_result = out_folder+"../Fathom_simulation.npy"
    fw_result = out_folder+"../Fathom_expected.npy"

    if emulator:
        checkExists(sim_result, 4)
    checkExists(graphFile)
    if (cpp==True):
        return graphFile, sim_result, fw_result, graphFile_mcm
    else:
        return graphFile, sim_result, fw_result

def execute_network(gf):
    print(header("Execute Network")) 
    checkExists(gf, 4)

    from shutil import copyfile

    # clean generated files
    generated_files = ["blob.bin", "vpu2.blob", "input.dat", "NCE2Task_network_out.bin", "output.dat", "expected_result_sim.dat"]
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

    command = "make run MV_SOC_REV=ma2490 MV_TOOLS_VERSION="+mvToolsV+" GRAPH_BIN=vpu2.blob GRAPH_BIN_PATH=. "+sPort + sIP 

    print(">>>>>>>>>>>>", command)
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT, cwd=APP_ROOT)

    if code.returncode != 0:
        print("Execution Failure")
        exit(code.returncode)

    result_file = APP_ROOT + "NCE2Task_network_out.bin"
    result_file_alt = [APP_ROOT + "NCE2Task0_out.bin", APP_ROOT + "output.dat"]  # Prod runtime uses "output.dat"

    result_file = checkExists(result_file, 3, aliases=result_file_alt)

    return result_file

def execute_network_mcm(gf):
    print(header("Execute Network")) 
    checkExists(gf, 4)

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

    print("Copy:", gf, "to", APP_ROOT + "blob.bin")
    copyfile(gf, APP_ROOT + "blob.bin")
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

    command = "make run MV_SOC_REV=ma2490 MV_TOOLS_VERSION="+mvToolsV+" GRAPH_BIN=blob.bin GRAPH_BIN_PATH=. "+sPort + sIP 

    print(">>>>>>>>>>>>", command)
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT, cwd=APP_ROOT)

    if code.returncode != 0:
        print("Execution Failure")
        exit(code.returncode)

    result_file = APP_ROOT + "NCE2Task_network_out.bin"
    result_file_alt = [APP_ROOT + "NCE2Task0_out.bin", APP_ROOT + "output.dat"]  # Prod runtime uses "output.dat"

    result_file = checkExists(result_file, 3, aliases=result_file_alt)

    return result_file

def validate_files(ref, test):

    checkExists(ref, 3)
    checkExists(test, 4)

    result = True

    print(header("Validate Result")) 

    command = "python3 "+POC_ROOT+"src2/Validate.py --reference "+ref+" --testdata "+ test + " --dtype u8"

    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
    
    if code.returncode != 0:
        result = False

    import zlib
    def crc(fileName):
        prev = 0
        for eachLine in open(fileName,"rb"):
            prev = zlib.crc32(eachLine, prev)
        return "%X"%(prev & 0xFFFFFFFF)

    print("CRC EXPECTED: ", crc(ref))
    print("CRC RESULT: ", crc(test))

    if crc(ref) == crc(test):
        # sys.exit(0)
        return 0
    else:
        return 5
        # sys.exit(5)