#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

TEST(quantization, case_conv)
{
    //Test based on res2a_branch2a/quantized_model.tflite modeil in POC
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64}, mv::DType("UInt8"), mv::Order("WHC"));

    mv::QuantizationParams inputQuantParams({128}, {0.00784314}, {0}, {1});
    input->set<mv::QuantizationParams>("quantizationParams", inputQuantParams);
    auto testShape = input->getShape();
    //EC: output defs are deduced from inputs

    mv::QuantizationParams outputQuantParams({128}, {0.00784314}, {0}, {1});
    mv::QuantizationParams weightsQuantParams({120}, {0.00272007}, {0}, {1});
    mv::QuantizationParams biasQuantParams({0}, {2.13339e-05}, {0}, {1});

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order(mv::Order::getColMajorID(4)), "weights");
    weights->set<mv::QuantizationParams>("quantizationParams", weightsQuantParams);
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1);
    auto convOp = om.getSourceOp(conv);
    std::vector<int64_t> biasesData = {
         4267,
         14962,
          7493,
        -13225,
        -11154,
         11836,
         -6836,
          6861,
          6515,
         10153,
         -2733,
         12968,
        -14782,
          9660,
          -411,
          8908,
         21138,
          7604,
         -7731,
         25183,
         -5499,
          3055,
         -1216,
          -534,
         -7641,
          1758,
          3676,
         19940,
         -8742,
         10501,
         -1248,
          7175,
         13393,
          4471,
         -1878,
        -10439,
          5313,
          2780,
          -290,
          1240,
          5604,
         -8112,
         -8254,
         15028,
         -3187,
         13932,
          7179,
          6903,
          2598,
          6988,
         12862,
          6600,
          3272,
         -4960,
          2603,
          3460,
          3936,
           779,
         -9498,
          1326,
         23844,
         18305,
         16151,
          7083,
    };
    mv::DataModel dm(om);
    auto biasTensor = dm.defineTensor("biasdata", {64}, mv::DType("Int32"), mv::Order(mv::Order::getColMajorID(1)), biasesData);
    om.addAttr(convOp, "bias", biasTensor->getName());
    biasTensor->set<mv::QuantizationParams>("quantizationParams", biasQuantParams);

    auto conv_output = convOp->getOutputTensor(0);
    conv_output->set<mv::QuantizationParams>("quantizationParams", outputQuantParams);

    mv::Element dummyPassDesc("dummyPassDesc");
    mv::json::Object compOutput;
    mv::TargetDescriptor desc;

    desc.setTarget(mv::Target::ma2490);

    mv::pass::PassRegistry::instance().find("MarkHardwareOperations")->run(om, desc, dummyPassDesc, compOutput);
    mv::pass::PassRegistry::instance().find("Quantization")->run(om, desc, dummyPassDesc, compOutput);

    //ref data is based on result on POC test res2a_branch2a/quantized_model.tflite
    std::vector<double> refData = {
        0 ,      0, 5832764,   51324,
        0 ,      0, 5832764,   62019,
        0 ,      0, 5832764,   54550,
        0 ,      0, 5832764,   33832,
        0 ,      0, 5832764,   35903,
        0 ,      0, 5832764,   58893,
        0 ,      0, 5832764,   40221,
        0 ,      0, 5832764,   53918,
        0 ,      0, 5832764,   53572,
        0 ,      0, 5832764,   57210,
        0 ,      0, 5832764,   44324,
        0 ,      0, 5832764,   60025,
        0 ,      0, 5832764,   32275,
        0 ,      0, 5832764,   56717,
        0 ,      0, 5832764,   46646,
        0 ,      0, 5832764,   55965,
        0 ,      0, 5832764,   68195,
        0 ,      0, 5832764,   54661,
        0 ,      0, 5832764,   39326,
        0 ,      0, 5832764,   72240,
        0 ,      0, 5832764,   41558,
        0 ,      0, 5832764,   50112,
        0 ,      0, 5832764,   45841,
        0 ,      0, 5832764,   46523,
        0 ,      0, 5832764,   39416,
        0 ,      0, 5832764,   48815,
        0 ,      0, 5832764,   50733,
        0 ,      0, 5832764,   66997,
        0 ,      0, 5832764,   38315,
        0 ,      0, 5832764,   57558,
        0 ,      0, 5832764,   45809,
        0 ,      0, 5832764,   54232,
        0 ,      0, 5832764,   60450,
        0 ,      0, 5832764,   51528,
        0 ,      0, 5832764,   45179,
        0 ,      0, 5832764,   36618,
        0 ,      0, 5832764,   52370,
        0 ,      0, 5832764,   49837,
        0 ,      0, 5832764,   46767,
        0 ,      0, 5832764,   48297,
        0 ,      0, 5832764,   52661,
        0 ,      0, 5832764,   38945,
        0 ,      0, 5832764,   38803,
        0 ,      0, 5832764,   62085,
        0 ,      0, 5832764,   43870,
        0 ,      0, 5832764,   60989,
        0 ,      0, 5832764,   54236,
        0 ,      0, 5832764,   53960,
        0 ,      0, 5832764,   49655,
        0 ,      0, 5832764,   54045,
        0 ,      0, 5832764,   59919,
        0 ,      0, 5832764,   53657,
        0 ,      0, 5832764,   50329,
        0 ,      0, 5832764,   42097,
        0 ,      0, 5832764,   49660,
        0 ,      0, 5832764,   50517,
        0 ,      0, 5832764,   50993,
        0 ,      0, 5832764,   47836,
        0 ,      0, 5832764,   37559,
        0 ,      0, 5832764,   48383,
        0 ,      0, 5832764,   70901,
        0 ,      0, 5832764,   65362,
        0 ,      0, 5832764,   63208,
        0 ,      0, 5832764,   54140
    };
    auto resData = dm.getTensor(convOp->get<std::string>("weightsTable"))->getIntData();
    for (unsigned i = 0; i < resData.size(); ++i)
       ASSERT_FLOAT_EQ(resData[i], refData[i]);
}
