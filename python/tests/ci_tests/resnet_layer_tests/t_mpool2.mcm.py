#!/usr/bin/python3

import sys, os
sys.path.append("..")

from test_helper import *

# skip()

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"
root_file = "mpool2"

model = generate_model(net_folder+root_file+"_MIG.csv")
_, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cmx=MAX_UINT32)
graphfile, _, _ = compile_graphFile(model, 4, 4, sch_folder+"SOH_POC.csv", emulator=False)

os.remove(graphfile)  # delete blob from previous run
graphfile, _, _ = compile_graphFile(model, 4, 4, sch_folder+"SOH_POC.csv", cpp=True)
t_result = execute_network(graphfile)
validate_files(s_result, t_result)