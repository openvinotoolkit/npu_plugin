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

static void build_Function_0(mv::OpModel& model)
{
    const auto input_0 = model.input("input", {13, 13, 1024, 1}, mv::DType("UInt8"), mv::Order("NHWC"), true);
    input_0->set<mv::QuantizationParams>("quantParams",  {{0},{1.0/255.0},{},{}});

    std::vector<int64_t> weights_conv1_data = mv::utils::generateSequence<int64_t> (3*3*1024*1024,255, 0); 
    const auto conv_weights_0 = model.constantInt("conv_weights", weights_conv1_data, {3, 3, 1024, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv_weights_0->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    
    const auto conv_0 = model.conv("conv", input_0, conv_weights_0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv_0->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});
    auto output = model.output("", conv_0, mv::DType("UInt8"), true);
}

TEST (mcmCompiler, streaming_performance)
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_Function_0(om);
    std::string compDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_kmb_with_CM_Conv.json";
    std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_kmb.json";

    unit.loadCompilationDescriptor(compDescPath);
    auto& mcmCompDesc = unit.compilationDescriptor();
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Info")));
    unit.loadTargetDescriptor(targetDescPath);
    unit.initialize();
    unit.run();
    
}

#endif // __GNUC__
