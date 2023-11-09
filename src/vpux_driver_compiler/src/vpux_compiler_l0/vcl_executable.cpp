//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vcl_executable.hpp"

using namespace vpux;

namespace VPUXDriverCompiler {
VPUXExecutableL0::VPUXExecutableL0(const NetworkDescription::Ptr& networkDesc, bool enableProfiling,
                                   VCLLogger* vclLogger)
        : _networkDesc(networkDesc), enableProfiling(enableProfiling), _logger(vclLogger) {
    _blob.clear();
}

vcl_result_t VPUXExecutableL0::serializeNetwork() {
    StopWatch stopWatch;
    if (enableProfiling) {
        stopWatch.start();
    }

    _blob = _networkDesc->getCompiledNetwork();

    if (enableProfiling) {
        stopWatch.stop();
        _logger->info("getCompiledNetwork time: {0} ms", stopWatch.delta_ms());
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXExecutableL0::getNetworkSize(uint64_t* blobSize) const {
    if (blobSize == nullptr) {
        _logger->outputError("Can not return blob size for NULL argument!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    *blobSize = _blob.size();
    if (*blobSize == 0) {
        // The executable handle do not contain a legal network.
        _logger->outputError("No blob created! The compiled network is empty!");
        return VCL_RESULT_ERROR_UNKNOWN;
    } else {
        return VCL_RESULT_SUCCESS;
    }
}

vcl_result_t VPUXExecutableL0::exportNetwork(uint8_t* blob, uint64_t blobSize) const {
    if (!blob || blobSize != _blob.size()) {
        _logger->outputError("Invalid argument to export network");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    StopWatch stopWatch;
    if (enableProfiling)
        stopWatch.start();

    memcpy(blob, _blob.data(), blobSize);

    if (enableProfiling) {
        stopWatch.stop();
        _logger->info("exportNetwork time: {0} ms", stopWatch.delta_ms());
    }
    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler
