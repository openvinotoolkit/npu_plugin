#!/usr/bin/python3

import sys, os
sys.path.append("..")
from test_helper import *

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"

root_file = "res3a_branch1"

model = generate_model(net_folder+root_file+"_MIG.csv")
graphfile, _, _ = compile_graphFile(model, 1, 1, sch_folder+"ResNet50_POC.csv", emulator=False)
_, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cmx=MAX_UINT32)

os.remove(graphfile)  # delete blob from previous run
_, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cmx=MAX_UINT32, cpp=True)
t_result = execute_network(graphfile)
validate_files(s_result, t_result)