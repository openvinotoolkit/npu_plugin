#! /usr/bin/env python3
from Controllers.Args import *
from Controllers.EnumController import throw_error, throw_warning, completion_msg
from Controllers.EnumController import *
from Controllers.Scheduler import load_network
from Models.EnumDeclarations import ErrorTable
from Environment import setup_hooks
import sys
import numpy as np
import os

if sys.version_info[0] != 3:
    sys.stdout.write("Attempting to run with a version of Python != 3.x\n")
    sys.exit(1)


major_version = np.uint32(3)
release_number = np.uint32(0)

# TODO: strip the .py from this file so we can run ./mcmFrontend

print("\033[1mmcmFrontend v" + (u"{0:02d}".format(major_version, )) + "." +
      (u"{0:02d}".format(release_number, )) +
      ", Copyright @ Movidius Ltd 2016\033[0m\n")

# TODO: Description of Modes


def main(args=None):

    setup_warnings()
    if args is None:
        args = define_and_parse_args()
    load_network(args)
if __name__ == "__main__":
    main()
