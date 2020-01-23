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

#include <hddl2_helpers/models/precompiled_resnet.h>
#include "hddl2_graph.h"

namespace vpu {
namespace HDDL2Plugin {

class ImportedGraph_Helper {
public:
    ImportedGraph_Helper();
    HDDL2Graph::Ptr getGraph();

protected:
    const std::string _modelToImport = PrecompiledResNet_Helper::resnet.graphPath;
    HDDL2Graph::Ptr _graphPtr = nullptr;
};

inline ImportedGraph_Helper::ImportedGraph_Helper() {
    _graphPtr= std::make_shared<HDDL2ImportedGraph>(_modelToImport);
}

HDDL2Graph::Ptr ImportedGraph_Helper::getGraph() {
    return _graphPtr;
}

}
}
