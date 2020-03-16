import os.path
import re

log_id = 0

regex = r"^\s+(\S+)$"
test_blobs = []
for match in [re.match(regex,line) for line in open("run_test.sh")]:
    if match:
        test_blobs.append(match.group(1))

test_id = 0
result = open("result.csv", "w")
while True:
    filename = "test_" + str(log_id) + ".log"
    if not os.path.isfile(filename):
        break

    regex = r".*?Median clock cycles: IR (\d+)"
    clockMatches = [re.match(regex, line) for line in open(filename)]
    IR = "NaN"
    for match in clockMatches:
        if match:
            IR = match.group(1)
            break

    regex = r".*?Instructions: (\d+)\s+Cycles: (\d+)\s+Branches: (\d+)\s+Stalls: (\d+)"
    perfCounterMatches = [re.match(regex, line) for line in open(filename)]
    perfCounters = ";".join(["NaN"] * 4)
    for match in perfCounterMatches:
        if match:
            perfCounters = ";".join([match.group(1), match.group(2), match.group(3), match.group(4)])
            break

    regex = r".*Test passed!.*"
    testPassedMatches = [re.match(regex, line) for line in open(filename)]
    passed = False
    for match in testPassedMatches:
        if match:
            passed = True
            break

    blob_info = ";".join(["NaN"] * 6)
    try:
        f = open("blobs_info/" + test_blobs[log_id] + ".csv", "r")
        blob_info = f.readline().replace("\n", "")
    except IOError:
        pass

    output = ";".join([blob_info, IR, perfCounters, "passed" if passed else "failed"])
    print(output)
    result.write(output + "\n")

    log_id += 1

