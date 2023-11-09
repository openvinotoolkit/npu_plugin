//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "ngraph_transformations.h"
#include <file_reader.h>
#include <ngraph/pass/serialize.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include "vpux/al/opset/opset_version.hpp"

namespace vpux {
namespace driverCompilerAdapter {
namespace ngraphTransformations {
const std::string opset = "opset";

static uint32_t opsetVersionToInt(const std::string& opsetVersion) {
    const std::size_t found = opsetVersion.find(opset);
    const std::string strVersion = opsetVersion.substr(found + opset.length());
    const uint32_t intVersion = tryStrToInt(strVersion);
    return intVersion;
}

bool isFunctionSupported(const std::shared_ptr<const ov::Model>& model, std::string opsetVersion) {
    std::string highestVersion = "opset0";
    for (const auto& op : model->get_ops()) {
        const std::string opVersionId = std::string(op->get_type_info().version_id);
        const uint32_t intVersion = opsetVersionToInt(opVersionId);
        const uint32_t largestIntVersion = opsetVersionToInt(highestVersion);
        if (intVersion > largestIntVersion) {
            highestVersion = opVersionId;
        }
    }

    const uint32_t highestIntVersion = opsetVersionToInt(highestVersion);
    const uint32_t intVersion = opsetVersionToInt(opsetVersion);

    if (highestIntVersion > intVersion) {
        return false;
    }
    return true;
}

void downgradeOpset(ngraph::pass::Manager& manager, const uint32_t& supportedVersionByCompiler) {
    // Extract the latest int version from the opset string version, i.e., opset11 -> 11
    uint32_t largestVersion = vpux::extractOpsetVersion();
    if (largestVersion > supportedVersionByCompiler) {
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }
}

IR serializeToIR(std::shared_ptr<ov::Model>& model, const uint32_t& supportedVersionByCompiler) {
    const auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager(passConfig);

    std::stringstream xmlStream, weightsStream;

    downgradeOpset(manager, supportedVersionByCompiler);

    manager.register_pass<ngraph::pass::Serialize>(xmlStream, weightsStream);
    manager.run_passes(model);
    const size_t xmlSize = vpu::KmbPlugin::utils::getFileSize(xmlStream);
    const size_t weightsSize = vpu::KmbPlugin::utils::getFileSize(weightsStream);
    std::vector<char> xmlBlob(xmlSize), weightsBlob(weightsSize);

    xmlStream.read(xmlBlob.data(), xmlSize);
    weightsStream.read(weightsBlob.data(), weightsSize);

    return {std::move(xmlBlob), std::move(weightsBlob)};
}

}  // namespace ngraphTransformations
}  // namespace driverCompilerAdapter
}  // namespace vpux
