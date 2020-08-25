import os
import re

regex = r"^\s+(\S+)$"
test_blobs = []
for match in [re.match(regex, line) for line in open("run_test.sh")]:
    if match:
        test_blobs.append(match.group(1))

logs = []
for filename in os.listdir("./"):
    match = re.match(r"test_(\d+).log", filename)
    if match:
        logs.append((int(match.group(1)), filename))

logs.sort()

result = open("result.csv", "w")
for log_id, filename in logs:
    regex = r".*?Median clock cycles: IR (\d+)"
    IR = "NaN"
    for match in [re.match(regex, line) for line in open(filename)]:
        if match:
            IR = match.group(1)
            break

    regex = r".*?Instructions: (\d+)\s+Cycles: (\d+)\s+Branches: (\d+)\s+Stalls: (\d+)"
    perfCounters = ";".join(["NaN"] * 4)
    for match in [re.match(regex, line) for line in open(filename)]:
        if match:
            perfCounters = ";".join(
                [match.group(1), match.group(2), match.group(3), match.group(4)])
            break

    regex = r".*Test passed!.*"
    passed = False
    for match in [re.match(regex, line) for line in open(filename)]:
        if match:
            passed = True
            break

    blob_info = ";".join(["NaN"] * 6)
    try:
        blobsInfoFilename = "blobs_info/" + test_blobs[log_id] + ".csv"
        regex = r"^(.*?;){5,}.*?$"
        for match in [re.match(regex, line) for line in open(blobsInfoFilename)]:
            if match:
                blob_info = match.group(0)
                break
    except IOError:
        pass

    output = ";".join([blob_info, IR, perfCounters,
                       "passed" if passed else "failed"])
    print(output)
    result.write(output + "\n")

    log_id += 1
