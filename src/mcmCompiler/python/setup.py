'''
TODO Write an actual setup script that will also call CMake build if necessary.
Currently this scripy only copies libs from build/ to api/ to enable execution.
'''

import os
import sys
import shutil

build_root = "../build/"
required_files = [("python/api/", "composition_api.py"), ("python/api/", "_composition_api.so"), ("lib/", "libcm.so")]

if not os.path.isdir(build_root):
    sys.exit("Build directory not found, build the project using CMake first")

for i in range(len(required_files)):
    if not os.path.exists(build_root + required_files[i][0] + required_files[i][1]):
        sys.exit("Required file " + build_root + str(required_files) + " not found, build the project using CMake first")
        
for i in range(len(required_files)):
    shutil.copy2(build_root + required_files[i][0] + required_files[i][1], "./api/" + required_files[i][1])