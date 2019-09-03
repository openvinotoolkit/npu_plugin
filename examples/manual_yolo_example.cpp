//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({6,6,512,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#20");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights0 = om.constantInt(weightsData0,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.003916202113032341},{-0.5246676802635193},{0.4739639163017273}}, "conv7_a_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv7_a#21");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{3.0715313187101856e-05},{-inf},{inf}}, "conv7_a_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights1 = om.constantInt(weightsData1,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.0038793424610048532},{-0.5122705698013306},{0.47696176171302795}}, "conv7_b_weights#4");
    auto conv1 = om.conv(input0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv7_b#22");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{3.0426215744228102e-05},{-inf},{inf}}, "conv7_b_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.125490203499794},{0.0},{32.0}});

    auto concat0 = om.concat({bias_c0, bias_c1}, "C", {{8},{0.12941177189350128},{-1.0352941751480103},{31.964706420898438}}, "concat/concat#23");

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*1024*256);
    auto weights2 = om.constantInt(weightsData2,{3,3,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.004127556923776865},{-0.5281267762184143},{0.5244002342224121}}, "conv8_a_weights#8");
    auto conv2 = om.conv(concat0, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv8_a#24");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005341544165275991},{-inf},{inf}}, "conv8_a_bias#9");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*1024*256);
    auto weights3 = om.constantInt(weightsData3,{3,3,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.00403307331725955},{-0.5408140420913696},{0.48761963844299316}}, "conv8_a_1_weights#14");
    auto conv3 = om.conv(concat0, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv8_a_1#26");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005219271406531334},{-inf},{inf}}, "conv8_a_1_bias#15");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (3*3*1024*256);
    auto weights4 = om.constantInt(weightsData4,{3,3,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.004036889877170324},{-0.4873768389225006},{0.5420300364494324}}, "conv8_b_weights#11");
    auto conv4 = om.conv(concat0, weights4, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv8_b#25");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005224210326559842},{-inf},{inf}}, "conv8_b_bias#12");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (3*3*1024*256);
    auto weights5 = om.constantInt(weightsData5,{3,3,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.003736605867743492},{-0.4958590269088745},{0.4569754898548126}}, "conv8_b_1_weights#17");
    auto conv5 = om.conv(concat0, weights5, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "conv8_b_1#27");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00048356078332290053},{-inf},{inf}}, "conv8_b_1_bias#18");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{0},{0.125490203499794},{0.0},{32.0}});

    auto concat1 = om.concat({bias_c2, bias_c4, bias_c3, bias_c5}, "C", {{8},{0.12941177189350128},{-1.0352941751480103},{31.964706420898438}}, "concat_1/FakeQuantWithMinMaxArgs#28");

    om.output(concat1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
