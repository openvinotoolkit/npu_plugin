//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <cpp_interfaces/exception2status.hpp>

namespace vpu {
namespace HDDL2Plugin {

#define HDDLUNITE_ERROR_str std::string("[HDDLUNITE_ERROR] ")
#define FILES_ERROR_str std::string("[FILES_ERROR] ")
#define CONFIG_ERROR_str std::string("[INVALID CONFIG] ")
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

///  Context
#define FAILED_CAST_CONTEXT               \
    std::string("Context is incorrect.\n" \
                "Please make sure you are using HDDL2RemoteContext.")

///  Infer request
#define NO_EXECUTOR_FOR_INFERENCE               \
    std::string("Can't create infer request!\n" \
                "Please make sure that the device is available. Only exports can be made.")

}  //  namespace HDDL2Plugin
}  //  namespace vpu
