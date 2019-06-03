//
// Copyright (C) 2019 Intel Corporation.
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

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>
#include <vpu/sw/utility.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(findSubGraphs);

    const auto& env = CompileEnv::get();

    int curSubGraphInd = 0;
    int curSubGraphSize = 0;

    for (const auto& stage : model->getStages()) {
        if (curSubGraphSize >= env.config.numberOfNodesInOneSubGraph) {
            curSubGraphSize = 0;
            curSubGraphInd++;
        }

        stage->attrs().set("subGraphInd", curSubGraphInd);
        curSubGraphSize++;
    }

    model->attrs().set("numSubGraphs", curSubGraphInd + 1);
}

}  // namespace

Pass::Ptr PassManager::findSubGraphs() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
