//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <memory>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include <ie_layers.h>

#include <vpu/graph_transformer.hpp>
#include <vpu/model/model.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class BackEnd final : public std::enable_shared_from_this<BackEnd> {
public:
    using Ptr = std::shared_ptr<BackEnd>;

    CompiledGraph::Ptr build(
            const Model::Ptr& model,
            const std::vector<ie::CNNLayerPtr>& allLayers);

    void dumpModel(
            const Model::Ptr& model,
            const std::string& postfix = std::string());

private:
    void serialize(
            const Model::Ptr& model,
            std::vector<char>& blob,
            std::pair<char*, size_t>& blobHeader,
            int& numActiveStages);

    void getMetaData(
            const Model::Ptr& model,
            const std::vector<ie::CNNLayerPtr>& allLayers,
            std::vector<StageMetaInfo>& metaData);

    void extractDataInfo(
            const Model::Ptr& model,
            DataInfo& inputInfo,
            DataInfo& outputInfo);

#ifndef NDEBUG
    void dumpModelToDot(
            const Model::Ptr& model,
            const std::string& fileName);
#endif
};

}  // namespace vpu
