// Copyright (C) 2019 Intel Corporation
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

#include <ie_blob.h>

#include <ie_input_info.hpp>
#include <ie_preprocess_data.hpp>
#include <map>
#include <string>

namespace InferenceEngine {
namespace KmbPreproc {

enum class Path : int { SIPP = 0, M2I, SHAVE_ONLY_M2I };

bool isApplicable(const BlobMap& inputs, const std::map<std::string, PreProcessDataPtr>& preprocData,
                  InputsDataMap& networkInputs);

void execDataPreprocessing(BlobMap& inputs, std::map<std::string, PreProcessDataPtr>& preprocData,
                           InputsDataMap& networkInputs, ColorFormat out_format, unsigned int numShaves,
                           unsigned int lpi, unsigned int numPipes, const std::string& preprocPoolId,
                           const int deviceId, Path path = Path::SIPP);

}  // namespace KmbPreproc
}  // namespace InferenceEngine
