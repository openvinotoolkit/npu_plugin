#!/usr/bin/python3

import sys
sys.path.append("..")
from test_helper import *
skip()

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"

root_file = "res2a_branch2c"

model = generate_model(net_folder+root_file+"_MIG.csv")
graphfile, s_result, _ = compile_graphFile(model, 4, 5, sch_folder+"SOH_POC.csv")

t_result = execute_network(graphfile)
validate_files(s_result, t_result)