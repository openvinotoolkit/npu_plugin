#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include <file_utils.h>

namespace {

class LinearizationTest : public ::testing::Test {
  protected:

    LinearizationTest() {}

    void LoadModel(mv::CompilationUnit* unit)
    {
        ASSERT_NE(unit, nullptr); // Validate the unit is not null.
        mv::OpModel& om = unit->model();

        unsigned short i0 = 0;
        unsigned short i1 = 1;
        unsigned short i3 = 3;
        unsigned short i4 = 4;
        unsigned short i56 = 56;
        unsigned short  i64 = 64;
        unsigned short  i118 = 118;
        unsigned short  i125 = 125;
        unsigned short  i128 = 128;

        float fM1 = -1.0;
        float f1 = 1.0;
        float f0 = 0.0;

        auto input0 = om.input("input#9", {i56,i56,i3,i1}, mv::DType("UInt8"), mv::Order::getZMajorID(i4));
        input0->setQuantParams({{i128},{0.007843137718737125},{fM1},{f1}});

        std::vector<int64_t> filterData0 = mv::utils::generateSequence<int64_t>(i3*i3*i3*i64);
        auto filter0 = om.constantInt("conv#0_filter#1", filterData0,{i3,i3,i3,i64}, mv::DType("UInt8"), mv::Order::getZMajorID(i4));
        filter0->setQuantParams({{135},{0.0025439101736992598},{-0.3435550332069397},{0.3051420748233795}});

        auto conv0 = om.conv("conv#10", input0, filter0, {i1, i1}, {i1, i1, i1, i1}, i1, i1);
        conv0->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t>(i64);
        mv::Data::TensorIterator biasWeights0 = om.constantInt("conv#0_bias#2", biasWeightsData0,{i64}, mv::DType("UInt8"), mv::Order::getColMajorID(i1));
        biasWeights0->setQuantParams({{i0},{1.9952236470999196e-05},{-inf_},{inf_}});
        auto bias_c0 = om.bias("", conv0, biasWeights0);
        bias_c0->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        std::vector<int64_t> filterData1 = mv::utils::generateSequence<int64_t> (i3*i3*i64*i128);
        auto filter1 = om.constantInt("conv_1#3_filter#4", filterData1,{i3,i3,i64,i128}, mv::DType("UInt8"), mv::Order::getZMajorID(i4));
        filter1->setQuantParams({{i125},{0.003295167814940214},{-0.41293057799339294},{0.4273372292518616}});
        auto conv1 = om.conv("conv_1#11", bias_c0, filter1, {i1, i1}, {i1, i1, i1, i1}, i1, i1);
        conv1->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (i128);
        auto biasWeights1 = om.constantInt("conv_1#3_bias#5", biasWeightsData1,{i128}, mv::DType("UInt8"), mv::Order::getColMajorID(i1));
        biasWeights1->setQuantParams({{i0},{1.292222714255331e-05},{-inf_},{inf_}});
        auto bias_c1 = om.bias("", conv1, biasWeights1);
        bias_c1->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        std::vector<int64_t> filterData2 = mv::utils::generateSequence<int64_t> (i3*i3*i128*i128);
        auto filter2 = om.constantInt("output#6_filter#7", filterData2,{i3,i3,i128,i128}, mv::DType("UInt8"), mv::Order::getZMajorID(i4));
        filter2->setQuantParams({{i118},{0.0037134578451514244},{-0.44002026319503784},{0.5069115161895752}});
        auto conv2 = om.conv("output#12", bias_c1, filter2, {i1, i1}, {i1, i1, i1, i1}, i1, i1);
        conv2->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (i128);
        auto biasWeights2 = om.constantInt("output#6_bias#8", biasWeightsData2,{i128}, mv::DType("UInt8"), mv::Order::getColMajorID(i1));
        biasWeights2->setQuantParams({{i0},{1.4562579963239841e-05},{-inf_},{inf_}});
        auto bias_c2 = om.bias("", conv2, biasWeights2);
        bias_c2->setQuantParams({{i0},{0.003921568859368563},{f0},{f1}});

        om.output("", bias_c2);
    }

    const float inf_ = std::numeric_limits<float>::infinity();
};

// Verify that all graphfile tensors are assigned sequential indices.
TEST_F(LinearizationTest, smoke)
{
  mv::CompilationUnit unit("parserModel");
  mv::OpModel& om = unit.model();
  std::string compDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_kmb_with_CM_Conv_hde.json";
  std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_kmb.json";
  LoadModel(&unit);
  unit.loadCompilationDescriptor(compDescPath);
  unit.loadTargetDescriptor(targetDescPath);
  unit.initialize();
  unit.run();
  mv::ControlModel controlModel(om);

  // Check the order of added operations.
  std::set<std::string> addedOps;
  for(auto op1 = controlModel.opBegin(); op1 != controlModel.opEnd(); ++op1)
  {
    if (op1->getName() == "ConstantInt_0_DDR2CMX")
    {
      ASSERT_NE(addedOps.find("conv_copyIn_0"), addedOps.end());
    }
    else if (op1->getName() == "conv_weights_table_DDR2CMX")
    {
      ASSERT_NE(addedOps.find("ConstantInt_0_DDR2CMX"), addedOps.end());
    }

    addedOps.emplace(op1->getName());
  }
}

}  // namespace
