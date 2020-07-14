// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <ie_input_info.hpp>
#include <ie_preprocess_data.hpp>
#include <map>
#include <string>

namespace InferenceEngine {
namespace KmbPreproc {

enum class Path : int { SIPP = 0, M2I };

bool useSIPP();
bool isApplicable(
    const BlobMap& inputs, const std::map<std::string, PreProcessDataPtr>& preprocData, InputsDataMap& networkInputs);

void execDataPreprocessing(BlobMap& inputs, std::map<std::string, PreProcessDataPtr>& preprocData,
    InputsDataMap& networkInputs, ColorFormat out_format, unsigned int numShaves, unsigned int lpi, const int& deviceId,
    Path path = Path::SIPP);

}  // namespace KmbPreproc
}  // namespace InferenceEngine
