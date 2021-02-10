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

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <file_utils.h>

// Create network:
// Input -> CONV1 -> CONV2(GroupConv) -> Output
void build_network_with_group_conv(mv::OpModel& model, std::size_t num_of_channels, std::size_t num_of_groups)
{
    mv::QuantizationParams quant_params_per_tensor({0}, {0.5}, {0.0}, {123.0}, {23}, {17123});
    std::vector<int64_t> zp_quant(num_of_channels, 0);
    std::vector<double> scale_quant(num_of_channels, 0.5);
    std::vector<double> min_quant(num_of_channels, 0.0);
    std::vector<double> max_quant(num_of_channels, 123.0);
    std::vector<unsigned> shift_quant(num_of_channels, 23);
    std::vector<unsigned> mult_quant(num_of_channels, 17123);
    mv::QuantizationParams quant_params_per_channel(zp_quant, scale_quant, min_quant, max_quant, shift_quant, mult_quant);

    const auto input = model.input("data", {64, 64, 3, 1}, mv::DType("UInt8"), mv::Order("NHWC"), true);
    input->set<mv::QuantizationParams>("quantParams", quant_params_per_tensor);

    std::vector<int64_t> weights_conv1_data = mv::utils::generateSequence<int64_t> (3*3*3*num_of_channels);
    auto weights_conv1 = model.constantInt("weights_conv1", weights_conv1_data, mv::Shape({3,3,3,num_of_channels}), mv::DType("UInt8"), mv::Order("NHWC"));
    weights_conv1->set<mv::QuantizationParams>("quantParams", quant_params_per_channel);

    const auto conv1 = model.conv("conv1", input, weights_conv1, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv1->set<mv::QuantizationParams>("quantParams", quant_params_per_tensor);

    std::vector<int64_t> weights_conv2_data = mv::utils::generateSequence<int64_t> (3*3*(num_of_channels/num_of_groups)*num_of_channels);
    auto weights_conv2 = model.constantInt("weights_conv2", weights_conv2_data, mv::Shape({3,3,(num_of_channels/num_of_groups),num_of_channels}), mv::DType("UInt8"), mv::Order("NHWC"));
    weights_conv2->set<mv::QuantizationParams>("quantParams", quant_params_per_channel);
    const auto conv2 = model.conv("conv2", conv1, weights_conv2, {1, 1}, {0, 0, 0, 0}, 1, num_of_groups);
    conv2->set<mv::QuantizationParams>("quantParams", quant_params_per_tensor);

    const auto output = model.output("", conv2, mv::DType("UInt8"), true);
}

template <typename T>
void verify_quant_params_num_for_slice_op(T input_quant, T output_quant, std::size_t num_of_groups)
{
    // If number of elements for quant parameter is more than 1
    // which is when quantization is per channel then slicing of
    // quant parameters should correspond to the number of groups
    // Otherwise (if 1 - quantization per tensor) the their number should match
    if (input_quant.size() > 1)
        ASSERT_EQ(input_quant.size()/num_of_groups, output_quant.size());
    else
        ASSERT_EQ(input_quant.size(), output_quant.size());
}

template <typename T>
void verify_quant_params_num(T quant_vec, std::size_t expected_num)
{
    auto quant_vec_size = quant_vec.size();
    if (quant_vec_size > 1)
        ASSERT_EQ(quant_vec_size, expected_num);
    else
        ASSERT_EQ(quant_vec_size, 1);
}

TEST (mcmCompiler, group_convolution_quant_param_test)
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    std::string compDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_kmb_with_CM_Conv.json";
    std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_kmb.json";

    std::size_t num_of_channels = 32;
    std::size_t num_of_groups = 2;

    build_network_with_group_conv(om, num_of_channels, num_of_groups);

    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(targetDescPath);
    unit.initialize();
    unit.run();

    // Get all slice conv operations
    std::vector<mv::Data::OpListIterator> convSlicesOps;
    std::vector<mv::Data::OpListIterator> sliceOps;
    std::vector<mv::Data::OpListIterator> concatOps;
    for (auto op = om.opBegin(); op != om.opEnd(); ++op)
    {
        auto opType = op->getOpType();
        if (opType == "DPUTask" && op->getName().find("slice") != std::string::npos)
        {
            convSlicesOps.push_back(op);
        }

        if (opType == "Slice" || opType == "ImplicitSlice")
            sliceOps.push_back(op);

        if (opType == "Concat" || opType == "ImplicitConcat")
            concatOps.push_back(op);
    }

    // Number of generated DPU taks operations for this GroupConv needs to match group value
    // Slices should be connected back with single concat
    ASSERT_EQ(convSlicesOps.size(), num_of_groups);
    ASSERT_EQ(sliceOps.size(), num_of_groups);
    ASSERT_EQ(concatOps.size(), 1);

    // Verify if slice operation correctly slices quant parameters
    for (auto& sliceOp : sliceOps)
    {
        auto input = sliceOp->getInputTensor(0);
        auto output = sliceOp->getOutputTensor(0);

        mv::QuantizationParams inputQuantParams = input->getQuantParams();
        mv::QuantizationParams outputQuantParams = output->getQuantParams();

        verify_quant_params_num_for_slice_op(inputQuantParams.getZeroPoint(), outputQuantParams.getZeroPoint(), num_of_groups);
        verify_quant_params_num_for_slice_op(inputQuantParams.getScale(), outputQuantParams.getScale(), num_of_groups);
        verify_quant_params_num_for_slice_op(inputQuantParams.getMin(), outputQuantParams.getMin(), num_of_groups);
        verify_quant_params_num_for_slice_op(inputQuantParams.getMax(), outputQuantParams.getMax(), num_of_groups);

        if (inputQuantParams.hasAttr("shift") && inputQuantParams.hasAttr("mult"))
        {
            verify_quant_params_num_for_slice_op(inputQuantParams.getShift(), outputQuantParams.getShift(), num_of_groups);
            verify_quant_params_num_for_slice_op(inputQuantParams.getMult(), outputQuantParams.getMult(), num_of_groups);
        }
    }

    // Verify if concat operation handles slices of computation correctly with respect to quant parameters
    {
        auto concatOp = concatOps[0];
        ASSERT_EQ(concatOp->get<std::string>("axis"), "C");

        auto inputs = concatOp->getInputTensor();
        auto output = concatOp->getOutputTensor(0);

        // Number of concat inputs should match the number of groups in GroupConvolution
        ASSERT_EQ(inputs.size(), num_of_groups);

        // Verify if concat operation inputs have correct size of quant params if they are per channel
        for (auto& input : inputs)
        {
            mv::QuantizationParams inputQuantParams = input->getQuantParams();
            verify_quant_params_num(inputQuantParams.getZeroPoint(), num_of_channels/num_of_groups);
            verify_quant_params_num(inputQuantParams.getScale(), num_of_channels/num_of_groups);
            verify_quant_params_num(inputQuantParams.getMin(), num_of_channels/num_of_groups);
            verify_quant_params_num(inputQuantParams.getMax(), num_of_channels/num_of_groups);
            if (inputQuantParams.hasAttr("shift") && inputQuantParams.hasAttr("mult"))
            {
                verify_quant_params_num(inputQuantParams.getShift(), num_of_channels/num_of_groups);
                verify_quant_params_num(inputQuantParams.getMult(), num_of_channels/num_of_groups);
            }
        }

        // Verify if concat operation correctly puts together quant params if they are per channel
        mv::QuantizationParams outputQuantParams = output->getQuantParams();
        verify_quant_params_num(outputQuantParams.getZeroPoint(), num_of_channels);
        verify_quant_params_num(outputQuantParams.getScale(), num_of_channels);
        verify_quant_params_num(outputQuantParams.getMin(), num_of_channels);
        verify_quant_params_num(outputQuantParams.getMax(), num_of_channels);
        if (outputQuantParams.hasAttr("shift") && outputQuantParams.hasAttr("mult"))
        {
            verify_quant_params_num(outputQuantParams.getShift(), num_of_channels);
            verify_quant_params_num(outputQuantParams.getMult(), num_of_channels);
        }
    }
}
