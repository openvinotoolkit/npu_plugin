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

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <models/precompiled_resnet.h>
#include "hddl2_graph.h"
#include <fstream>

namespace vpu {
namespace HDDL2Plugin {

class ImportedGraph_Helper {
public:
    ImportedGraph_Helper();
    Graph::Ptr getGraph();
    static void skipMagic(std::ifstream &blobStream);

protected:
    const std::string _modelToImport = PrecompiledResNet_Helper::resnet50.graphPath;
    Graph::Ptr _graphPtr = nullptr;
};

inline void ImportedGraph_Helper::skipMagic(std::ifstream &blobStream) {
    if (!blobStream.is_open()) {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << NETWORK_NOT_READ;
    }
    InferenceEngine::ExportMagic magic = {};
    blobStream.seekg(0, blobStream.beg);
    blobStream.read(magic.data(), magic.size());
    auto exportedWithName = (exportMagic == magic);
    if (exportedWithName) {
        std::string tmp;
        std::getline(blobStream, tmp);
    } else {
        blobStream.seekg(0, blobStream.beg);
    }
}

inline ImportedGraph_Helper::ImportedGraph_Helper() {
    MCMConfig defaultConfig;
    std::ifstream blobFile(_modelToImport, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << NETWORK_NOT_READ;
    }

    skipMagic(blobFile);
    _graphPtr= std::make_shared<ImportedGraph>(blobFile, defaultConfig);
}

inline Graph::Ptr ImportedGraph_Helper::getGraph() {
    return _graphPtr;
}

}
}
