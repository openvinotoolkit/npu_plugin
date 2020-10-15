import sys
import os

base = os.environ.get('MCM_HOME')
ldlib = os.environ.get('LD_LIBRARY_PATH')
if base is None:
    print("Please set environment path MCM_HOME. Exiting...")
    quit()
if ldlib is None:
    print("Please set your LD_LIBRARY_PATH environment variable correctly. Exiting...")
    quit()

sys.path.append(base + '/python/api')
