#!/usr/bin/python3

import sys
sys.path.append("..")
from test_helper import *

skip()

# Not Configured

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"

root_file = "test_2layer_maxpool"

model = generate_model(net_folder+root_file+"_MIG.csv", network=True)
graphfile, s_result, _ = compile_graphFile(model, 4, 4, os.path.join(p, "../../../config/resnet50_strategy.csv"))

t_result = execute_network(graphfile)
validate_files(s_result, t_result)