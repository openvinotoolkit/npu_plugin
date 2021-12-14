//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifdef _WIN32
#include <direct.h>
#define getcwd _getcwd
#elif
#include <unistd.h>
#endif
#include <stdlib.h>
#include <regex>

#include "test_model/kmb_test_base.hpp"

#include <hetero/hetero_plugin_config.hpp>
#include <hetero_vpux_sqlite.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>
#include "vpux_private_config.hpp"

PRETTY_PARAM(Device, std::string)
PRETTY_PARAM(NetworkPath, std::string)
PRETTY_PARAM(SplitLayer, std::string)
PRETTY_PARAM(ImagePath, std::string)

using HeteroPluginSplitNetworkParams = std::tuple<NetworkPath, ImagePath>;

static int64_t time_int64() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}

/**
* Iterates over the layers network, trying to split over each layer, persists results into sqlite db.
* Actual test run is performed by the separate process (which can segfault or reboot the host)
*/
class HeteroPluginSplitNetworkTest :
        public testing::WithParamInterface<HeteroPluginSplitNetworkParams>,
        public KmbClassifyNetworkTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HeteroPluginSplitNetworkParams>& obj);
};

std::string HeteroPluginSplitNetworkTest::getTestCaseName(
        const testing::TestParamInfo<HeteroPluginSplitNetworkParams>& obj) {
    const std::string network = std::get<0>(obj.param);

    std::stringstream testName;
    testName << network << "_testcase";
    return testName.str();
}

TEST_P(HeteroPluginSplitNetworkTest, splitOverAllLayers) {
    SKIP_ON("HDDL2", "Stability problems");

    const auto envVarSplitNetworkName = std::getenv("IE_KMB_HETERO_TESTS_SPLIT_NETWORK");
    if (envVarSplitNetworkName == nullptr) {
        std::cout << "IE_KMB_HETERO_TESTS_SPLIT_NETWORK is not set. Skippping" << std::endl;
        GTEST_SKIP() << "IE_KMB_HETERO_TESTS_SPLIT_NETWORK is not set. Skippping";
    }

    const auto network = envVarSplitNetworkName;

    std::vector<std::string> layers;
    try {
        TestNetworkDesc netDesc(network);
        std::cout << "Reading network to get list of layers: " << netDesc.irFileName() << std::endl;
        auto networkCNN = readNetwork(netDesc, true);
        auto networkFunc = networkCNN.getFunction();
        if (!networkFunc) {
            std::cerr << "Empty network function!" << std::endl;
            FAIL();
        }
        std::cout << "Collecting list of layers... ";
        auto orderedOps = networkFunc->get_ordered_ops();
        std::transform(orderedOps.cbegin(), orderedOps.cend(), std::back_inserter(layers),
                       [](auto node) -> std::string {
                           return node->get_friendly_name();
                       });
    } catch (const std::exception& e) {
        std::cerr << "Exception caught while reading the network: " << e.what()
                  << std::endl;
        FAIL();
    }
    std::cout << "Done. Network contains " << layers.size() << " layers..." << std::endl;

    const std::string envNetworkValue = std::string("IE_KMB_HETERO_TESTS_SPLIT_NETWORK=") + std::string(network);
    if (putenv(envNetworkValue.c_str()) != 0) {
        std::cerr << "Can not set environment variable value" << std::endl;
        FAIL();
    }

    const size_t delim = envNetworkValue.find_last_of("/");
    const std::string networkFileName = (delim == std::string::npos) ? "" : envNetworkValue.substr(delim + 1);
    const size_t dot = networkFileName.find(".");
    const std::string networkShortName = (dot == std::string::npos) ? "" : networkFileName.substr(0, dot);
    const std::string networkAlphanumName = std::regex_replace(networkShortName, std::regex("[^a-zA-Z0-9]"), "_");
    const std::string databaseName = "hetero_splits.db";

    char cwd[256];
    std::cout << "Current working dir is " << getcwd(cwd, sizeof(cwd)) << std::endl;
    SqliteSupport sqlite(databaseName, networkAlphanumName);
    std::string lastLayer;
    int64_t startTime, finishTime;
    inferStateEnum inferState;
    const bool lastLayerExists = sqlite.getLastLayerStarted(lastLayer, startTime, finishTime, inferState);
    if (lastLayerExists) {
        std::cout << "Last persisted split layer '" << lastLayer << "', status " << (int)inferState << std::endl;

        if (inferState == inferStateEnum::TO_BE_RUN) {
            // finish up last record before reboot
            if (finishTime == 0) {
                const int64_t nowtime = time_int64();
                sqlite.updateLayer(startTime, nowtime, inferStateEnum::FAIL_REBOOT);
                std::cout << "Last persisted split layer was '" << lastLayer << "' with no finish time; set updated status to rebooted "
                          << (int)inferStateEnum::FAIL_REBOOT << std::endl;
            } else {
                std::cerr << "Invalid sqlite database state: last to be run record has finish time " << std::endl;
                FAIL();
            }
        }
    } else {
        std::cout << "No last persisted split layer found, starting with the first from network " << std::endl;
    }

    char curDir[512];
    if (curDir != getcwd(curDir, sizeof(curDir))) {
        std::cerr << "Can not determine current dir " << std::endl;
        FAIL();
    }

    char exeName[512];
    char* exePath;
    if (_get_pgmptr(&exePath) != 0) {
        std::cerr << "Can not determine current exe name " << std::endl;
        FAIL();
    }

    std::vector<std::string> splitSuccessfully;
    const auto layersCount = layers.size();
    size_t lid = 0;
    bool rewinding = lastLayerExists;
    for (auto&& splitNode : layers) {
        ++lid;
        if (rewinding) {
            std::cout << "Rewinding the layer '" << splitNode << "' until layer '" << lastLayer << "' (" << lid
                      << " of " << layersCount << ")" << std::endl;

            if (splitNode == lastLayer) {
                rewinding = false;
                std::cout << "Rewinded last layer '" << splitNode << "' (" << lid << " of " << layersCount << ")"
                          << std::endl;
            }
            continue;
        }

        try {
            std::string envLayerValue = "IE_KMB_HETERO_TESTS_SPLIT_LAYER=" + splitNode;
            if (putenv(envLayerValue.c_str()) != 0) {
                std::cerr << "Can not set environment variable value" << std::endl;
                break;
            }

            const int64_t nowtime1 = time_int64();
            sqlite.insertLayer(lid, splitNode, nowtime1);
            sqlite.flush();
            const std::string cmdline = std::string(exePath) + 
                                        " --gtest_filter=*environment*HeteroPluginTest*envNetworkEnvLayerSplit* " +
                                        splitNode;
            std::cout << "Exec " << cmdline << std::endl;
            int ret = system(cmdline.c_str());

            const int64_t nowtime2 = time_int64();
            if (ret == 0) {
                splitSuccessfully.push_back(splitNode);
                sqlite.updateLayer(nowtime1, nowtime2, inferStateEnum::INFERRED_OK);
            } else {
                sqlite.updateLayer(nowtime1, nowtime2, inferStateEnum::FAIL_SIGSEG);
            }
            std::cout << "Returned " << ret << " on single layer '" << splitNode << "' (" << lid << " of "
                      << layersCount << "), " << splitSuccessfully.size() << " successful splits " << std::endl;

        } catch (const std::exception& e) {
            const int64_t nowtime2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count();
            sqlite.updateLayer(time_int64(), nowtime2, inferStateEnum::FAIL_EXCEPTION);
            std::cout << "Exception caught while iterating through the layer '" << splitNode << "': " << e.what()
                      << std::endl;
        }
    }

    std::cout << "Found " << splitSuccessfully.size() << " splits without exception: " << std::endl;
    std::for_each(splitSuccessfully.cbegin(), splitSuccessfully.cend(), [](std::string split) {
        std::cout << split << "; ";
    });
}

// test running search of working splits for single network; network filename is passed via environment variable
INSTANTIATE_TEST_CASE_P(
        envNetworkSplit, HeteroPluginSplitNetworkTest,
        ::testing::Combine(
                ::testing::Values(NetworkPath(
                        "env var")),
                ::testing::Values(ImagePath("unused"))),
        HeteroPluginSplitNetworkTest::getTestCaseName);
