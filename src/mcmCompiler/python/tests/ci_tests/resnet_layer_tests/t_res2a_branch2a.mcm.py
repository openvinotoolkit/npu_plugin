#!/usr/bin/python3

import sys, os
sys.path.append("..")
from test_helper import *
CWD = os.getcwd() + "/"

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"
root_file = "res2a_branch2a"

model = generate_model(net_folder+root_file+"_MIG.csv")
graphfile_poc, s_result, _, graphfile_mcm = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cpp=True)
result_list = []
t_mcm_result = execute_network_mcm(graphfile_mcm)
result_list.append(validate_files(s_result, t_mcm_result))

if (sys.argv[1] == "poc"):
    t_poc_result = execute_network(graphfile_poc)
    result_list.append(validate_files(s_result, t_poc_result))

if (sys.argv[2] == "compare"):
    command = 'cd '+CWD+'"/output";'+'flatc -t "$GRAPHFILE/src/schema/graphfile.fbs" -- blob.bin;'+ \
        'flatc -t "$GRAPHFILE/src/schema/graphfile.fbs" -- vpu2.blob;'+'meld vpu2.json blob.json;'
    code = subprocess.run(command, shell=True, stderr=subprocess.STDOUT)

sys.exit(result_list)