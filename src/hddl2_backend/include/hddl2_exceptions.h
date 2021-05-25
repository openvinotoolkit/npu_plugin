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

#pragma once

namespace vpux {
namespace hddl2 {

#define HDDLUNITE_ERROR_str std::string("[HDDLUNITE_ERROR] ")
#define FILES_ERROR_str std::string("[FILES_ERROR] ")
#define CONFIG_ERROR_str std::string("[INVALID CONFIG] ")
#define PARAMS_ERROR_str std::string("[INVALID PARAMS] ")
#define CONTEXT_ERROR_str std::string("[INVALID REMOTE CONTEXT] ")

///  Scheduler
#define FAILED_START_SERVICE                                     \
    std::string("Couldn't start the device scheduler service.\n" \
                "Please start the service or check the environment variable \"KMB_INSTALL_DIR\".")
#define SERVICE_AVAILABLE std::string("HDDL Scheduler service is available. Ready to go.")
#define SERVICE_NOT_AVAILABLE \
    std::string("HDDL Scheduler service is not available. Please start scheduler before running application.")

///  Network
#define FAILED_LOAD_NETWORK                                   \
    std::string("Couldn't load the graph into the device.\n"  \
                "Please check the service logs for errors.\n" \
                "A reboot may be required to restore the device to a functional state.")

/// Executor
#define EXECUTOR_NOT_CREATED                                                              \
    std::string("No executor has been created for the device, only export is possible.\n" \
                "For execution, please start the service or check the environment variable \"KMB_INSTALL_DIR\".")

///  Infer request
#define NO_EXECUTOR_FOR_INFERENCE               \
    std::string("Can't create infer request!\n" \
                "Please make sure that the device is available. Only exports can be made.")

}  //  namespace hddl2
}  //  namespace vpux
