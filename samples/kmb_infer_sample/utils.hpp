//
// Copyright 2019 Intel Corporation.
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

#include <samples/common.hpp>

#include <inference_engine.hpp>

namespace IE = InferenceEngine;

std::vector<std::string> readLabelsFromFile(const std::string& labelFileName);

IE::Blob::Ptr preprocessUncompiledLayers(const std::string &layersPath, const std::string& data);
IE::Blob::Ptr postprocessUncompiledLayers(const std::string &layersPath, IE::Blob::Ptr src);

