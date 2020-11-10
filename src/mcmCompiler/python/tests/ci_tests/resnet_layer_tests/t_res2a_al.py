#!/usr/bin/python3

import sys
sys.path.append("..")
from test_helper import *

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"
root_file = "res2a_branch2a"

model = generate_model(net_folder+root_file+"_MIG.csv")
graphfile, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv")
try:
    os.remove(graphfile)
except FileNotFoundError:
    pass

graphfile, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cpp=True)

t_result = execute_network(graphfile)
validate_files(s_result, t_result)

#**Error Code Guide**:
# 0 - Success
# 1 - General Failure
# 2 - Reserved.
# 3 - Compiler Failed to generate graphfile
# 4 - Runtime Failed to generate NCETask0.bin
# 5 - Correctness Error
#
# 123 - Skipped Test
# 127 - Timeout