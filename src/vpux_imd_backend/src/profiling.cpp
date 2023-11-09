//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/profiling.hpp"
#include "vpux/IMD/infer_request.hpp"
#include "vpux/utils/plugin/profiling_parser.hpp"

#include <fstream>
#include <vector>

namespace vpux {
namespace IMD {

using namespace vpux::profiling;

LayerStatistics getLayerStatistics(const uint8_t* rawData, size_t dataSize, const std::vector<char>& blob) {
    ProfilingFormat format = ProfilingFormat::NONE;
    std::ofstream outFile = openProfilingStream(&format);

    const uint8_t* blob_data = reinterpret_cast<const uint8_t*>(blob.data());
    std::vector<LayerInfo> layerProfiling;

    if (outFile.is_open()) {
        if (format == ProfilingFormat::RAW) {
            saveRawDataToFile(rawData, dataSize, outFile);
        } else {
            std::vector<TaskInfo> taskProfiling =
                    getTaskInfo(blob_data, blob.size(), rawData, dataSize, TaskType::ALL, VerbosityLevel::HIGH);
            layerProfiling = getLayerInfo(blob_data, blob.size(), rawData, dataSize);
            saveProfilingDataToFile(format, outFile, layerProfiling, taskProfiling);
        }
    } else {
        layerProfiling = getLayerInfo(blob_data, blob.size(), rawData, dataSize);
    }

    return convertLayersToIeProfilingInfo(layerProfiling);
}

}  // namespace IMD
}  // namespace vpux
