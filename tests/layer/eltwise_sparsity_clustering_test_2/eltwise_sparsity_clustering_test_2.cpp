// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"
#include "eltwise_sparsity_clustering_test_2.data.inc"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const auto input_7_0 = model.input({32, 32, 16, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{128},{0.007843137718737},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input#7");
    const auto conv1a_0_weights_1_0 = model.constantInt(conv1a_0_weights_1_0_data, {3, 3, 16, 32}, mv::DType("UInt8"), mv::Order("NCHW"), {{115},{0.002828093012795},{-0.325303435325623},{0.395860284566879},{0},{1}}, "conv1a#0_weights#1");
    const auto conv1a_8_0 = model.conv(input_7_0, conv1a_0_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv1a#8");
    const auto conv1a_0_bias_2weights_0 = model.constantInt(conv1a_0_bias_2weights_0_data, {32}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000022181122404},{-inf},{inf},{0},{1}}, "conv1a#0_bias#2weights");
    const auto conv1a_0_bias_2_0 = model.bias(conv1a_8_0, conv1a_0_bias_2weights_0, mv::DType("UInt8"), {{0},{0.000022181122404},{-inf},{inf},{0},{1}}, "conv1a#0_bias#2");
    const auto conv1b_3_weights_4_0 = model.constantInt(conv1b_3_weights_4_0_data, {3, 3, 16, 32}, mv::DType("UInt8"), mv::Order("NCHW"), {{128},{0.002850869437680},{-0.365644007921219},{0.361327707767487},{0},{1}}, "conv1b#3_weights#4");
    const auto conv1b_9_0 = model.conv(input_7_0, conv1b_3_weights_4_0, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv1b#9");
    const auto conv1b_3_bias_5weights_0 = model.constantInt(conv1b_3_bias_5weights_0_data, {32}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000022359759896},{-inf},{inf},{0},{1}}, "conv1b#3_bias#5weights");
    const auto conv1b_3_bias_5_0 = model.bias(conv1b_9_0, conv1b_3_bias_5weights_0, mv::DType("UInt8"), {{0},{0.000022359759896},{-inf},{inf},{0},{1}}, "conv1b#3_bias#5");
    const auto eltwise_FakeQuantWithMinMaxArgs_10_0 = model.eltwise({conv1a_0_bias_2_0, conv1b_3_bias_5_0}, "Add", mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "eltwise/FakeQuantWithMinMaxArgs#10");
    const auto output = model.output(eltwise_FakeQuantWithMinMaxArgs_10_0, mv::DType("Default"), {{},{},{},{}}, true, "");
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

