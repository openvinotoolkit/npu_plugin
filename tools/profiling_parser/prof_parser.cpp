//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ie_version.hpp>
#include <iostream>

#include <gflags/gflags.h>

#include <flatbuffers/minireflect.h>

#include <schema/profiling_generated.h>

#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/plugin/profiling_meta.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

using vpux::profiling::OutputType;
using vpux::profiling::VerbosityLevel;

DEFINE_string(b, "", "Precompiled blob that was profiled");
DEFINE_string(p, "", "Profiling result binary");
DEFINE_string(f, "json", "Format to use (text, json or debug)");
DEFINE_string(o, "", "Output file, stdout by default");
DEFINE_bool(g, false, "Profiling data is from FPGA");
DEFINE_bool(v, false, "Increased verbosity of DPU tasks parsing (include variant level tasks)");
DEFINE_bool(vv, false, "Highest verbosity of tasks parsing (Currently same as -v)");
DEFINE_bool(m, false, "Dump profiling metadata");

static bool validateFile(const char* flagName, const std::string& pathToFile) {
    if (pathToFile.empty()) {
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
    if (FLAGS_vv) {
        return VerbosityLevel::HIGH;
    } else if (FLAGS_v) {
        return VerbosityLevel::MEDIUM;
    } else {
        return VerbosityLevel::LOW;
    }
}

static std::string verbosityToStr(VerbosityLevel verbosity) {
    std::map<VerbosityLevel, std::string> labels = {
            {VerbosityLevel::LOW, "Low"},
            {VerbosityLevel::MEDIUM, "Medium"},
            {VerbosityLevel::HIGH, "High"},
    };
    return labels[verbosity];
}

static OutputType getOutputFormat() {
    if (FLAGS_f == "text") {
        return OutputType::TEXT;
    } else if (FLAGS_f == "json") {
        return OutputType::JSON;
    } else if (FLAGS_f == "debug") {
        return OutputType::DEBUG;
    }
    VPUX_THROW("Unknown output format: {0}. Valid formats: text, json", FLAGS_f);
}

static void parseCommandLine(int argc, char* argv[], const std::string& usage) {
    gflags::SetUsageMessage(usage);
    gflags::RegisterFlagValidator(&FLAGS_b, &validateFile);
    gflags::SetVersionString(InferenceEngine::GetInferenceEngineVersion()->buildNumber);

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (!FLAGS_m && !validateFile("-p", FLAGS_p)) {
        throw std::runtime_error("Invalid -p parameter value");
    }
}

static void printCommandLineParameters() {
    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network blob file:     " << FLAGS_b << std::endl;
    std::cout << "    Profiling result file: " << FLAGS_p << std::endl;
    std::cout << "    Format (text/json):    " << FLAGS_f << std::endl;
    std::cout << "    Output file:           " << FLAGS_o << std::endl;
    std::cout << "    Verbosity:             " << verbosityToStr(getVerbosity()) << std::endl;
    std::cout << "    FPGA:                  " << FLAGS_g << std::endl;
    std::cout << "    Dump metadata:         " << FLAGS_m << std::endl;
    std::cout << std::endl;
}

static void dumpProfilingMetadata(const uint8_t* blobData, size_t blobSize) {
    const uint8_t* sectionData = vpux::profiling::getProfilingSectionPtr(blobData, blobSize);

    const auto prettyProfilingMeta =
            flatbuffers::FlatBufferToString(sectionData, ProfilingFB::ProfilingMetaTypeTable(), /*multi_line*/ true,
                                            /*vector_delimited*/ false);
    std::cout << prettyProfilingMeta << std::endl;
}

int main(int argc, char** argv) {
    static const char* usage = "Usage: prof_parser -b <blob path> -p <profiling.bin path> [-f json|text] "
                               "[-o <output.file>] [-v|vv] [-g] [-m]";
    try {
        parseCommandLine(argc, argv, usage);
        printCommandLineParameters();

        std::string blobPath(FLAGS_b);
        std::ifstream blob_file;
        blob_file.open(blobPath, std::ios::in | std::ios::binary);
        blob_file.seekg(0, blob_file.end);
        size_t blob_length = blob_file.tellg();
        blob_file.seekg(0, blob_file.beg);
        std::vector<uint8_t> blob_bin(blob_length);
        blob_file.read(reinterpret_cast<char*>(blob_bin.data()), blob_bin.size());
        blob_file.close();

        if (FLAGS_m) {
            if (!FLAGS_o.empty() || !FLAGS_p.empty()) {
                throw std::runtime_error("-o|-p parameters have no effect when -m is specified");
            }

            dumpProfilingMetadata(blob_bin.data(), blob_bin.size());
            return 0;
        }

        std::string profResult(FLAGS_p);
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
        auto format = getOutputFormat();

        vpux::profiling::outputWriter(format, blobData, profilingData, FLAGS_o, getVerbosity(), FLAGS_g);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << usage << std::endl;
        return 1;
    }

    return 0;
}
