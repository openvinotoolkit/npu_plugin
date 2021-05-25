//
// Copyright 2020 Intel Corporation.
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

// TODO: Temporarly skip this test for Ubuntu20 with gcc9 due to internal
// GCC compiler error (LTO) happening on CI infrastructure
#if __GNUC__ != 9

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <limits>
#include <file_utils.h>


static void build_Function_1(mv::OpModel& model)
{
    using namespace mv;
    static const double max_inf = std::numeric_limits<double>::infinity();
    static const double min_inf = -std::numeric_limits<double>::infinity();

    const auto input_0 = model.input("input", {16, 16, 256, 1}, mv::DType("BFloat16"), mv::Order("NHWC"), true);
    input_0->set<mv::QuantizationParams>("quantParams", {{0},{1},{min_inf},{max_inf},{0},{1}});
    std::vector<int64_t> weights_conv1_data = mv::utils::generateSequence<int64_t> (256* 64);
    const auto conv_weights_0 = model.constantInt("conv_weights", weights_conv1_data, {1, 1, 256, 64}, mv::DType("BFloat16"), mv::Order("NCHW"));
    conv_weights_0->set<mv::QuantizationParams>("quantParams", {{0},{1},{min_inf},{max_inf},{0},{1}});
    const auto conv_0 = model.conv("conv", input_0, conv_weights_0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv_0->set<mv::QuantizationParams>("quantParams", {{0},{1},{min_inf},{max_inf},{0},{1}});
    const auto output = model.output("", conv_0, mv::DType("BFloat16"), true);
}

TEST (mcmCompiler, mtl_conv_bf16)
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_Function_1(om);

    std::string compDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_mtl-sc.json";
    std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_mtl.json";

    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(targetDescPath);
    unit.initialize();
    unit.run();

    std::vector<mv::Data::OpListIterator> convops;
    for (auto op = om.opBegin(); op != om.opEnd(); ++op)
    {
        auto opType = op->getOpType();
        if ((opType == "DPUTask") && op->get<std::string>("taskOp") == "Conv")
        {
            convops.push_back(op);
        }
    }
    ASSERT_EQ(convops.size(), 1);
    auto bfloatDType = mv::DType("BFloat16");
    ASSERT_TRUE(convops[0]->getInputTensor(0)->getDType() == bfloatDType);
    ASSERT_TRUE(convops[0]->getInputTensor(1)->getDType() == bfloatDType);
    ASSERT_TRUE(convops[0]->getOutputTensor(0)->getDType() == bfloatDType);
    ASSERT_TRUE(convops[0]->hasAttr("weightsTableIndex"));
    ASSERT_TRUE(convops[0]->hasAttr("floatScale"));

    auto floatScale = convops[0]->get<std::vector<float>>("floatScale");
    for (auto a = floatScale.begin(); a != floatScale.end(); a++)
        ASSERT_EQ(*a , 1.0);
}

#endif // __GNUC__
