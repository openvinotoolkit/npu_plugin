// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "kmb_test_tool.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsUtils {

class KmbLayerTestsCommon : virtual public LayerTestsCommon {
protected:
    KmbTestTool kmbTestTool;
public:
    explicit KmbLayerTestsCommon();
    void Run() override;
protected:
    void BuildNetworkWithoutCompile();
    void ImportNetwork();
    void ExportNetwork();

    void Validate() override;
    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;
};

extern const TargetDevice testPlatformTargetDevice;

}  // namespace LayerTestsUtils
