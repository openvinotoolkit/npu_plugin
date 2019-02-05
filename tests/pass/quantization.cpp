#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

TEST(quantization, case_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64}, mv::DType("UInt8"), mv::Order("WHC"));

    mv::QuantizationParams inputQuantParams({128}, {0.00784314}, {0}, {1});
    input->set<mv::QuantizationParams>("quantizationParams", inputQuantParams);

    //auto output = om.output({56, 56, 64}, mv::DType("UInt8"), mv::Order("WHC"));
    //output->set<mv::QuantizationParams>("quantizationParams", outputQuantParams);
    //EC: output defs are deduced from inputs

    mv::QuantizationParams outputQuantParams({128}, {0.00784314}, {0}, {1});

/////////////////
    std::vector<double> weightsData = mv::utils::generateSequence<double>(64*64);
    auto weights = om.constant(weightsData, {64, 64, 1, 1}, mv::DType("UInt8"), mv::Order(mv::Order::getColMajorID(4)), "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1, outputQuantParams);
    auto convOp = om.getSourceOp(conv);
    std::vector<double> biasesData = mv::utils::generateSequence<double>(32);
    auto biases = om.constant(biasesData, {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)), "biases");
    auto bias = om.bias(conv, biases);
    auto biasOp = om.getSourceOp(bias);
    om.output(bias);

    auto outputOp = biasOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("QuantizationPass")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    //Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 4);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);

    for (unsigned i = 0; i < dm.getTensor(convOp->get<std::string>("bias"))->getData().size(); ++i)
        ASSERT_FLOAT_EQ(dm.getTensor(convOp->get<std::string>("bias"))->getData()[i], biasesData[i]);

}
