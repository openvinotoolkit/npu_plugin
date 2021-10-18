//
// Copyright Intel Corporation.
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

#include "ngraph_transformations.h"
#include <file_reader.h>
#include <ngraph/pass/manager.hpp>
#include <transformations/serialize.hpp>
#include <chrono>

namespace vpux {
namespace zeroCompilerAdapter {
namespace ngraphTransformations {

IR serializeToIR(const std::shared_ptr<ngraph::Function>& netGraph) {
    const std::unique_ptr<vpu::Logger> _logger = std::unique_ptr<vpu::Logger>(
        new vpu::Logger("ngraphTransformations", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()));

    using ms = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();

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

    auto finish = std::chrono::high_resolution_clock::now();
    _logger->info("|| Timer ||;ngraphTransformations::serializeToIR (ms);\t{}", std::chrono::duration_cast<ms>(finish - start).count());
    return {xmlBlob, weightsBlob};
}

}  // namespace ngraphTransformations
}  // namespace zeroCompilerAdapter
}  // namespace vpux
