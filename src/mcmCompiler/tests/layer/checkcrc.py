import sys, os
import subprocess
import csv
import re 
def validate_files(ref, test):
# dump first few lines of the results files
    refBF = open(ref, "rb")
    testBF = open(test, "rb")
    for f in [ref, test]:
        print("Partial dump of ", f)
        with open(f, "rb") as bf:
            for i in range(0, 3):  # lines to dump
                print(" ".join("{:02X}".format(x) for x in bf.read(32)))  # number of bytes to dump/line
    import zlib
    def crc(fileName):
        prev = 0
        for eachLine in open(fileName,"rb"):
            prev = zlib.crc32(eachLine, prev)
        return "%X"%(prev & 0xFFFFFFFF)
    print("CRC EXPECTED: ", crc(ref))
    print("CRC RESULT: ", crc(test))
    if crc(ref) == crc(test):
        print("PASS")
        sys.exit(1)
    else:
        print("FAIL")
        sys.exit(0)
    return result
validate_files("expected_result_sim.dat", "output-0.bin")
