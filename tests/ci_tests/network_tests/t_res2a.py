#!/usr/bin/python3

import sys
sys.path.append("..")
from test_helper import *

skip()

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"

root_file = "test_res2a"

model = generate_model(net_folder+root_file+".py", network=True)
graphfile, _, _ = compile_graphFile(model, 4, 5, sch_folder+"ResNet50_POC.csv", emulator=False)
_, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cmx=MAX_UINT32)

t_result = execute_network(graphfile)
validate_files(s_result, t_result)