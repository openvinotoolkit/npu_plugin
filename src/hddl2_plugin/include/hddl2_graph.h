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

#include <map>
#include <memory>
#include <string>

#include "mcm_network_description.hpp"

// TODO This part should be replaced with vpux::Compiler (MCMCompiler)
namespace vpu {
namespace HDDL2Plugin {
namespace Graph {
vpux::NetworkDescription::Ptr compileGraph(InferenceEngine::ICNNNetwork& network, const MCMConfig& config);

vpux::NetworkDescription::Ptr importGraph(const std::string& blobFilename, const MCMConfig& config);

vpux::NetworkDescription::Ptr importGraph(std::istream& networkModel, const MCMConfig& config);
}  // namespace Graph
}  // namespace HDDL2Plugin
}  // namespace vpu
