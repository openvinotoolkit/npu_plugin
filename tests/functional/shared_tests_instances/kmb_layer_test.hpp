// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "kmb_test_env_cfg.hpp"
#include "kmb_test_tool.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsUtils {

class KmbLayerTestsCommon : virtual public LayerTestsCommon {
protected:
    static const KmbTestEnvConfig envConfig;
    KmbTestTool kmbTestTool;

public:
    explicit KmbLayerTestsCommon();
    void Run() override;

protected:
    void BuildNetworkWithoutCompile();
    virtual void ImportNetwork();
    void ExportNetwork();
    void ImportInput();
    void ExportInput();
    void ExportOutput();
    void ImportReference(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs);
    void ExportReference(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs);

    void Validate() override;

    virtual void SkipBeforeLoad() {
    }
    virtual void SkipBeforeImport() {
    }
    virtual void SkipBeforeInfer() {
    }
    virtual void SkipBeforeValidate() {
    }

    void useCompilerMLIR();
    void setReferenceSoftwareModeMLIR();
    void setReferenceHardwareModeMLIR();
    bool isCompilerMCM() const;
    bool isCompilerMLIR() const;

    void disableMcmPasses(const std::vector<std::pair<std::string, std::string>>& banList);

    void TearDown() override;
};

class KmbSkipTestException : public std::runtime_error {
public:
    KmbSkipTestException(const std::string& what_arg): runtime_error(what_arg){};
};

extern const TargetDevice testPlatformTargetDevice;

}  // namespace LayerTestsUtils
