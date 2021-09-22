//
// Copyright Intel Corporation.
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

#include "zero_compiler_adapter.h"
#include <gtest/gtest.h>
#include <ngraph/output_vector.hpp>
#include "icompiler_adapter.h"
#include "ngraph/ngraph.hpp"
#include "ngraph_transformations.h"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include "ie_ngraph_utils.hpp"
#include "ie_core.hpp"

namespace vpux {
namespace zeroCompilerAdapter {

namespace IE = InferenceEngine;

using NgraphChecker = std::function<void(const std::shared_ptr<const ngraph::Function>& netGraph)>;
class CompilerInDriverStub : public ICompiler_Adapter {
public:
    using Ptr = std::shared_ptr<CompilerInDriverStub>;

    explicit CompilerInDriverStub(const Opset& opsetVersion, NgraphChecker ngraphChecker): _opsetVersion(opsetVersion), _ngraphChecker(ngraphChecker) {}

    Opset getSupportedOpset() override {
        return _opsetVersion;
    };

private:
    /**
     * @brief Check that ngraph provided to compiler in driver meets the requirements
     * @details compileIR should be called after all ngraph transformation,
     *  we can convert xml + bin back to ngraph function and use any checks we need
     */
    Blob::Ptr compileIR(std::vector<char>& xml, std::vector<char>& weights) override {
        InferenceEngine::Core ie;
        const std::string xmlString(xml.begin(), xml.end());

        IE::TensorDesc tensorDesc(IE::Precision::U8, {1, weights.size()}, IE::Layout::NC);
        const auto weightsBlob = IE::make_shared_blob<uint8_t>(tensorDesc);
        weightsBlob->allocate();
        const auto rawBlob = weightsBlob->buffer().as<float*>();
        ie_memcpy(rawBlob, weightsBlob->byteSize(), weights.data(), weights.size());

        const auto network = ie.ReadNetwork(xmlString, weightsBlob);
        const auto function = network.getFunction();

        // Validate
        _ngraphChecker(function);

        // Return empty blob, anyway will not be used
        return std::make_shared<Blob>(std::vector<char>());
    }
    std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap> getNetworkMeta(
            const Blob::Ptr ) override {
        return std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap>();
    }
    std::tuple<const DataMap, const DataMap> getDeviceNetworkMeta(const Blob::Ptr ) override {
        return std::tuple<const DataMap, const DataMap>();
    }


private:
    const Opset _opsetVersion;
    NgraphChecker _ngraphChecker;
};

//------------------------------------------------------------------------------
class ZeroCompilerAdapter_UnitTests : public ::testing::Test {};

//------------------------------------------------------------------------------
TEST_F(ZeroCompilerAdapter_UnitTests, CompilerOpset4_OperationOpset6_LoweringWillBeApplied) {
    // Prepare compiler stub, which will pretend, that only opset 4 supported and will throw otherwise
    const Opset compilerOpsetVersion = {4};
    // Already have such check after lowering call, but duplicate in case it will be removed
    static NgraphChecker opsetCheck = [&](const std::shared_ptr<const ngraph::Function>& netGraph) -> void
    {
        bool isSupported = ngraphTransformations::isFunctionSupported(netGraph, compilerOpsetVersion);
        if (!isSupported) {
            THROW_IE_EXCEPTION << "Not supported version!";
        }
    };
    CompilerInDriverStub::Ptr compilerInDriverStub = std::make_shared<CompilerInDriverStub>(compilerOpsetVersion, opsetCheck);
    std::shared_ptr<vpux::ICompiler> compiler = std::make_shared<ZeroCompilerAdapter>(compilerInDriverStub);

    // Prepare ngraph function with opset 6
    const auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
    const auto axesConst = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
    const auto mvn6 = std::make_shared<ngraph::opset6::MVN>(
            data, axesConst, false, 1e-5, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

    std::shared_ptr<ngraph::Function> opset6mvn = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn6 }, ngraph::ParameterVector{ data });

    ASSERT_NO_THROW(compiler->compile(opset6mvn, "mvn6", InferenceEngine::InputsDataMap(), InferenceEngine::OutputsDataMap()));
}

}
}
