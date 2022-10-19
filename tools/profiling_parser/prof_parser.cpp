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
using vpux::profiling::TimeUnitFormat;
using vpux::profiling::VerbosityLevel;

DEFINE_string(b, "", "Precompiled blob that was profiled.");
DEFINE_string(p, "", "Profiling result binary");
DEFINE_string(f, "json", "Format to use (text or json)");
DEFINE_string(o, "", "Output file, stdout by default");
DEFINE_string(u, "ms", "Precision format for TraceEventFormat(ns or ms), ms by default");
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
    std::cout << "    Time unit:             " << FLAGS_u << std::endl;
    std::cout << "    Verbosity:             " << verbosityToStr[getVerbosity()] << std::endl;

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const std::string usage = "Usage: prof_parser -b <blob path> -p <output.bin path> [-f json|text] [-o "
                              "<output.file>] [-u ns|ms] [-v|vv]\n";
    if (argc < 5) {
        std::cout << usage << std::endl;
        return 0;
    }
    parseCommandLine(argc, argv, usage);

    const std::string blobPath(FLAGS_b);
    const std::string profResult(FLAGS_p);
    std::transform(FLAGS_f.begin(), FLAGS_f.end(), FLAGS_f.begin(), ::tolower);
    const OutputType format = (FLAGS_f == "text") ? OutputType::TEXT : OutputType::JSON;
    std::transform(FLAGS_u.begin(), FLAGS_u.end(), FLAGS_u.begin(), ::tolower);
    const TimeUnitFormat timeFormat = (FLAGS_u == "ns") ? TimeUnitFormat::NS : TimeUnitFormat::MS;

    std::ifstream blob_file;
    blob_file.open(blobPath, std::ios::in | std::ios::binary);
    blob_file.seekg(0, blob_file.end);
    const int blob_length = static_cast<int>(blob_file.tellg());
    blob_file.seekg(0, blob_file.beg);
    std::vector<char> blob_bin(blob_length);
    blob_file.read(blob_bin.data(), blob_length);
    blob_file.close();

    std::fstream profiling_results;
    profiling_results.open(profResult, std::ios::in | std::ios::binary);
    profiling_results.seekg(0, profiling_results.end);
    const int profiling_length = static_cast<int>(profiling_results.tellg());
    profiling_results.seekg(0, profiling_results.beg);
    std::vector<char> output_bin(profiling_length);
    profiling_results.read(output_bin.data(), profiling_length);
    profiling_results.close();

    const auto blobData = std::make_pair(reinterpret_cast<uint8_t*>(blob_bin.data()), blob_length);
    const auto profilingData = std::make_pair(reinterpret_cast<uint8_t*>(output_bin.data()), profiling_length);

    vpux::profiling::outputWriter(format, blobData, profilingData, FLAGS_o, timeFormat, getVerbosity());

    return 0;
}
