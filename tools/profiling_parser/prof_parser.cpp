//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ie_version.hpp>
#include <iostream>

#include <gflags/gflags.h>

#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

using vpux::profiling::OutputType;
using vpux::profiling::VerbosityLevel;

DEFINE_string(b, "", "Precompiled blob that was profiled.");
DEFINE_string(p, "", "Profiling result binary");
DEFINE_string(f, "json", "Format to use (text, json or debug)");
DEFINE_string(o, "", "Output file, stdout by default");
DEFINE_bool(g, false, "Profiling data is from FPGA");
DEFINE_bool(v, false, "Medium verbosity of DPU tasks parsing");
DEFINE_bool(vv, false, "High verbosity of DPU tasks parsing");

static bool validateFile(const char* flagName, const std::string& pathToFile) {
    if (pathToFile.empty()) {
        // Default value must fail validation
        return false;
    }
    std::ifstream ifile;
    ifile.open(pathToFile);

    const bool isValid = ifile.good();
    if (isValid) {
        ifile.close();
    } else {
        std::cerr << "Got error when parsing argument \"" << flagName << "\" with value " << pathToFile << std::endl;
    }
    return isValid;
}

static VerbosityLevel getVerbosity() {
    VerbosityLevel verbosity = FLAGS_v == true ? VerbosityLevel::MEDIUM : VerbosityLevel::LOW;
    return FLAGS_vv == true ? VerbosityLevel::HIGH : verbosity;
}

static void parseCommandLine(int argc, char* argv[], const std::string& usage) {
    gflags::SetUsageMessage(usage);
    gflags::RegisterFlagValidator(&FLAGS_b, &validateFile);
    gflags::RegisterFlagValidator(&FLAGS_p, &validateFile);

    std::ostringstream version;
    version << InferenceEngine::GetInferenceEngineVersion();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::map<VerbosityLevel, std::string> verbosityToStr = {
            {VerbosityLevel::LOW, "Low"},
            {VerbosityLevel::MEDIUM, "Medium"},
            {VerbosityLevel::HIGH, "High"},
    };

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network blob file:     " << FLAGS_b << std::endl;
    std::cout << "    Profiling result file: " << FLAGS_p << std::endl;
    std::cout << "    Format (text/json):    " << FLAGS_f << std::endl;
    std::cout << "    Output file:           " << FLAGS_o << std::endl;
    std::cout << "    Verbosity:             " << verbosityToStr[getVerbosity()] << std::endl;
    std::cout << "    FPGA:                  " << FLAGS_g << std::endl;

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    static const char* usage = "Usage: prof_parser -b <blob path> -p <profiling.bin path> [-f json|text] "
                               "[-o <output.file>] [-v|vv] [-g]\n";
    if (argc < 5) {
        std::cout << usage << std::endl;
        return 0;
    }
    parseCommandLine(argc, argv, usage);

    std::string blobPath(FLAGS_b);
    std::string profResult(FLAGS_p);
    std::transform(FLAGS_f.begin(), FLAGS_f.end(), FLAGS_f.begin(), ::tolower);
    OutputType format =
            (FLAGS_f == "text") ? OutputType::TEXT : ((FLAGS_f == "json") ? OutputType::JSON : OutputType::DEBUG);

    std::ifstream blob_file;
    blob_file.open(blobPath, std::ios::in | std::ios::binary);
    blob_file.seekg(0, blob_file.end);
    size_t blob_length = blob_file.tellg();
    blob_file.seekg(0, blob_file.beg);
    std::vector<uint8_t> blob_bin(blob_length);
    blob_file.read(reinterpret_cast<char*>(blob_bin.data()), blob_bin.size());
    blob_file.close();

    std::fstream profiling_results;
    profiling_results.open(profResult, std::ios::in | std::ios::binary);
    profiling_results.seekg(0, profiling_results.end);
    size_t profiling_length = profiling_results.tellg();
    profiling_results.seekg(0, profiling_results.beg);
    std::vector<uint8_t> output_bin(profiling_length);
    profiling_results.read(reinterpret_cast<char*>(output_bin.data()), output_bin.size());
    profiling_results.close();

    auto blobData = std::make_pair(blob_bin.data(), blob_bin.size());
    auto profilingData = std::make_pair(output_bin.data(), output_bin.size());

    vpux::profiling::outputWriter(format, blobData, profilingData, FLAGS_o, getVerbosity(), FLAGS_g);

    return 0;
}
