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

#define FAILED_START_SCHEDULER                                   \
    std::string("Couldn't start the device scheduler service.\n" \
                "Please start the service or check the environment variable \"KMB_INSTALL_DIR\".")
#define GRAPH_NOT_LOADED                                                         \
    std::string("Graph was not loaded on the device, only export can be made.\n" \
                "For execution, please start the service or check the environment variable \"KMB_INSTALL_DIR\".")

}  //  namespace HDDL2Plugin
}  //  namespace vpu
