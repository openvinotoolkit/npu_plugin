//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "ngraph_transformations.h"
#include <file_reader.h>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/serialize.hpp>

namespace vpux {
namespace driverCompilerAdapter {
namespace ngraphTransformations {

bool isFunctionSupported(const std::shared_ptr<const ngraph::Function>& netGraph, size_t opsetVersion) {
    size_t highestVersion = 0;
    for (const auto& op : netGraph->get_ops()) {
        if (op->get_version() > highestVersion) {
            highestVersion = op->get_version();
        }
    }

    if (highestVersion > opsetVersion) {
        return false;
    }
    return true;
}

IR serializeToIR(const std::shared_ptr<ngraph::Function>& netGraph) {
    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager(passConfig);

    std::stringstream xmlStream, weightsStream;
    manager.register_pass<ngraph::pass::Serialize>(xmlStream, weightsStream);
    manager.run_passes(netGraph);

    const size_t xmlSize = vpu::KmbPlugin::utils::getFileSize(xmlStream);
    const size_t weightsSize = vpu::KmbPlugin::utils::getFileSize(weightsStream);
    std::vector<char> xmlBlob(xmlSize), weightsBlob(weightsSize);

    xmlStream.read(xmlBlob.data(), xmlSize);
    weightsStream.read(weightsBlob.data(), weightsSize);

    return {xmlBlob, weightsBlob};
}

}  // namespace ngraphTransformations
}  // namespace driverCompilerAdapter
}  // namespace vpux
