//This file is the parsed network which is created through python.
#include <unistd.h>

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"


struct InputParams {

  InputParams() : comp_descriptor_(NULL) {}

  bool parse_args(int argc, char **argv) {
    int opt;
    char const * const options = "d:";

    while ((opt = getopt(argc, argv, options)) != -1) {
      switch (opt) {
        case 'd':
          comp_descriptor_ = optarg;
          break;
        default:
          usage();
          return false;
      }
    }

    if (!comp_descriptor_) { 
      usage();
      return false; 
    }
    return true;
  }

  void usage() const {
    fprintf(stderr, "./three_layer_conv_model -d {comp_descriptor}\n");
  }

  const char *comp_descriptor_;
};  // struct InputParams //

int main(int argc, char **argv)
{

    InputParams params;

    if (!params.parse_args(argc, argv)) { return -1; }

    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");

    ////////////////////////////////////////////////////////////////////////////
      mv::OpModel& om = unit.model();
      auto input0 = om.input("input#9", {56,56,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
      input0->setQuantParams({{128},{0.007843137718737125},{-1.0},{1.0}});

      std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*64);
      auto weights0 = om.constantInt("conv#0_weights#1", weightsData0,{3,3,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
      weights0->setQuantParams({{135},{0.0025439101736992598},{-0.3435550332069397},{0.3051420748233795}});

      auto conv0 = om.conv("conv#10", input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1);
      conv0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
      auto biasWeights0 = om.constantInt("conv#0_bias#2", biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
      auto bias_c0 = om.bias("", conv0, biasWeights0);
      biasWeights0->setQuantParams({{0},{1.9952236470999196e-05},{-inf},{inf}});
      bias_c0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*64*128);
      auto weights1 = om.constantInt("conv_1#3_weights#4", weightsData1,{3,3,64,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
      auto conv1 = om.conv("conv_1#11", bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1);
      weights1->setQuantParams({{125},{0.003295167814940214},{-0.41293057799339294},{0.4273372292518616}});
      conv1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> biasWeightsData1 =
          mv::utils::generateSequence<int64_t> (128);
      auto biasWeights1 = om.constantInt("conv_1#3_bias#5", biasWeightsData1,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
      auto bias_c1 = om.bias("", conv1, biasWeights1);
      biasWeights1->setQuantParams({{0},{1.292222714255331e-05},{-inf},{inf}});
      bias_c1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*128*128);
      auto weights2 = om.constantInt("output#6_weights#7", weightsData2,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
      auto conv2 = om.conv("output#12", bias_c1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1);
      weights2->setQuantParams({{118},{0.0037134578451514244},{-0.44002026319503784},{0.5069115161895752}});
      conv2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (128);
      auto biasWeights2 = om.constantInt("output#6_bias#8", biasWeightsData2,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
      auto bias_c2 = om.bias("", conv2, biasWeights2);
      biasWeights2->setQuantParams({{0},{1.4562579963239841e-05},{-inf},{inf}});
      bias_c2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

      om.output("", bias_c2);
    ////////////////////////////////////////////////////////////////////////////


    std::string compDescPath = params.comp_descriptor_;
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
