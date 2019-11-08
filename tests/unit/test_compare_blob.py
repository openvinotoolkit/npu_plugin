# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

# utility for comparing json files, based on existing code from 
# https://stackoverflow.com/questions/25851183/how-to-compare-two-json-objects-with-the-same-elements-in-a-different-order-equa

import sys, getopt, os
from pathlib import Path
import subprocess
import json

def compare_json_lists(list1, list2):
    if (list1.__len__() != list2.__len__()):
        return False

    for l in list1:
        found = False
        for m in list2:
            res = compare_json_lists(l, m)
            if (res):
                found = True
                break

        if (not found):
            return False

    return True

def compare_json_objects(obj1, obj2):
    if isinstance(obj1, list):
        if (not isinstance(obj2, list)):
            return False
        return compare_json_lists(obj1, obj2)
    elif (isinstance(obj1, dict)):
        if (not isinstance(obj2, dict)):
            return False
        exp = set(obj2.keys()) == set(obj1.keys())
        if (not exp):
            print("Differences: ", obj1.keys(), obj2.keys())
            return False
        for k in obj1.keys():
            val1 = obj1.get(k)
            val2 = obj2.get(k)
            if isinstance(val1, list):
                if (not compare_json_lists(val1, val2)):
                    return False
            elif isinstance(val1, dict):
                if (not compare_json_objects(val1, val2)):
                    return False
            else:
                if val2 != val1:
                    return False
    else:
        return obj1 == obj2

    return True


def main(argv):
    # parse input arguments
    expected_blob = ''
    actual_blob = ''
    try:
        opts, args = getopt.getopt(argv,"ha:e:",["actual=","expected="])
    except getopt.GetoptError:
        print ('test_compare_blob.py -a <actual_blob> -e <expected_blob>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('test_compare_blob.py -a <actual_blob> -e <expected_blob>')
            sys.exit()
        elif opt in ("-a", "--actual"):
            actual_blob = arg
        elif opt in ("-e", "--expected"):
            expected_blob = arg

    # check file exists
    actual_exists = Path(actual_blob)
    if not actual_exists.is_file():
        print ('Actual Blob file does not exist: ', actual_blob)
        sys.exit(2)
    
    expected_exists = Path(expected_blob)
    if not expected_exists.is_file():
        print ('Expected Blob file does not exist: ', expected_blob)
        sys.exit(2)
    
    print ('Actual blob:  ', actual_blob)
    print ('Expected blob: ', expected_blob)

    #convert blob to json
    subprocess.run(["flatc", "-t", os.path.expandvars("$GRAPHFILE/src/schema/graphfile.fbs"), "--strict-json", "--", actual_blob])
    subprocess.run(["flatc", "-t", os.path.expandvars("$GRAPHFILE/src/schema/graphfile.fbs"), "--strict-json", "--", expected_blob])

    # find output files
    tmp_arr = os.path.basename(actual_blob).split(".")
    actual_file = tmp_arr[0] + ".json"
    tmp_arr = os.path.basename(expected_blob).split(".")
    expected_file = tmp_arr[0] + ".json"

    # do the comparison
    print("Comparing: " + actual_file)
    print(expected_file)
    with open(actual_file, "r") as read_file:
        actual_json = json.load(read_file)
    with open(expected_file, "r") as read_file:
        expected_json = json.load(read_file)

    result = compare_json_objects(actual_json, expected_json)
    print (result)

if __name__ == "__main__":
   main(sys.argv[1:])
