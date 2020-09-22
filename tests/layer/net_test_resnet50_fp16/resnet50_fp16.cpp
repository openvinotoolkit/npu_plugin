#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

#include "resnet50_fp16.hpp"

template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)
{
    std::ifstream file(input_file, std::ifstream::binary);
    T2 inputString;
    std::vector<T2> data;
    while(file.read(reinterpret_cast<char*>(&inputString), sizeof(T2)))
        data.push_back(inputString);
    file.close();
    std::vector<T1> return_data(data.begin(), data.end());
    return return_data;
}

mv::CompilationUnit buildResnet50_fp16(const std::string& binaryDir)
{
    using std::int32_t;
    using std::int64_t;
    using std::uint8_t;

    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit compilationUnit("resnet50_fp16");
    mv::OpModel& om = compilationUnit.model();

    auto input0 = om.input({224,224,3,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input#234");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/conv1#0_weights#1.dat");
    auto weights0 = om.constantInt(weightsData0,{7,7,3,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "conv1#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {2, 3, 2, 3}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "conv1#307");

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/conv1#0_bias#2.dat");
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "conv1#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu0 = om.relu(bias_c0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "conv1#235");

    auto pool0 = om.maxPool(relu0, {3, 3}, {2, 2}, {0, 1, 0, 1}, true, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "pool1/max_pool#236");

    std::vector<int64_t> weightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch1#5_weights#6.dat");
    auto weights1 = om.constantInt(weightsData1,{1,1,64,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2a_branch1#5_weights#6");
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch1#308");

    std::vector<int64_t> biasWeightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch1#5_bias#7.dat");
    auto biasWeights1 = om.constantInt(biasWeightsData1,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2a_branch1#5_bias#7");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu1 = om.relu(bias_c1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch1#237");

    std::vector<int64_t> weightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2a#9_weights#10.dat");
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2a_branch2a#9_weights#10");
    auto conv2 = om.conv(pool0, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2a#309");

    std::vector<int64_t> biasWeightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2a#9_bias#11.dat");
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2a_branch2a#9_bias#11");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu2 = om.relu(bias_c2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2a#238");

    std::vector<int64_t> weightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2b#13_weights#14.dat");
    auto weights3 = om.constantInt(weightsData3,{3,3,64,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2a_branch2b#13_weights#14");
    auto conv3 = om.conv(relu2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2b#310");

    std::vector<int64_t> biasWeightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2b#13_bias#15.dat");
    auto biasWeights3 = om.constantInt(biasWeightsData3,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2a_branch2b#13_bias#15");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu3 = om.relu(bias_c3, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2b#239");

    std::vector<int64_t> weightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2c#17_weights#18.dat");
    auto weights4 = om.constantInt(weightsData4,{1,1,64,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2a_branch2c#17_weights#18");
    auto conv4 = om.conv(relu3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2c#311");

    std::vector<int64_t> biasWeightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2a_branch2c#17_bias#19.dat");
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2a_branch2c#17_bias#19");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu4 = om.relu(bias_c4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a_branch2c#240");

    auto eltwise0 = om.eltwise({relu1,relu4}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2a/Add#241");

    std::vector<int64_t> weightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2a#22_weights#23.dat");
    auto weights5 = om.constantInt(weightsData5,{1,1,256,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2b_branch2a#22_weights#23");
    auto conv5 = om.conv(eltwise0, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2a#312");

    std::vector<int64_t> biasWeightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2a#22_bias#24.dat");
    auto biasWeights5 = om.constantInt(biasWeightsData5,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2b_branch2a#22_bias#24");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu5 = om.relu(bias_c5, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2a#242");

    std::vector<int64_t> weightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2b#26_weights#27.dat");
    auto weights6 = om.constantInt(weightsData6,{3,3,64,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2b_branch2b#26_weights#27");
    auto conv6 = om.conv(relu5, weights6, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2b#313");

    std::vector<int64_t> biasWeightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2b#26_bias#28.dat");
    auto biasWeights6 = om.constantInt(biasWeightsData6,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2b_branch2b#26_bias#28");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu6 = om.relu(bias_c6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2b#243");

    std::vector<int64_t> weightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2c#30_weights#31.dat");
    auto weights7 = om.constantInt(weightsData7,{1,1,64,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2b_branch2c#30_weights#31");
    auto conv7 = om.conv(relu6, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2c#314");

    std::vector<int64_t> biasWeightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2b_branch2c#30_bias#32.dat");
    auto biasWeights7 = om.constantInt(biasWeightsData7,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2b_branch2c#30_bias#32");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu7 = om.relu(bias_c7, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b_branch2c#244");

    auto eltwise1 = om.eltwise({eltwise0,relu7}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2b/Add#245");

    std::vector<int64_t> weightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2a#35_weights#36.dat");
    auto weights8 = om.constantInt(weightsData8,{1,1,256,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2c_branch2a#35_weights#36");
    auto conv8 = om.conv(eltwise1, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2a#315");

    std::vector<int64_t> biasWeightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2a#35_bias#37.dat");
    auto biasWeights8 = om.constantInt(biasWeightsData8,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2c_branch2a#35_bias#37");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu8 = om.relu(bias_c8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2a#246");

    std::vector<int64_t> weightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2b#39_weights#40.dat");
    auto weights9 = om.constantInt(weightsData9,{3,3,64,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2c_branch2b#39_weights#40");
    auto conv9 = om.conv(relu8, weights9, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2b#316");

    std::vector<int64_t> biasWeightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2b#39_bias#41.dat");
    auto biasWeights9 = om.constantInt(biasWeightsData9,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2c_branch2b#39_bias#41");
    auto bias_c9 = om.bias(conv9, biasWeights9, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu9 = om.relu(bias_c9, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2b#247");

    std::vector<int64_t> weightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2c#43_weights#44.dat");
    auto weights10 = om.constantInt(weightsData10,{1,1,64,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res2c_branch2c#43_weights#44");
    auto conv10 = om.conv(relu9, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2c#317");

    std::vector<int64_t> biasWeightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res2c_branch2c#43_bias#45.dat");
    auto biasWeights10 = om.constantInt(biasWeightsData10,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res2c_branch2c#43_bias#45");
    auto bias_c10 = om.bias(conv10, biasWeights10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu10 = om.relu(bias_c10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c_branch2c#248");

    auto eltwise2 = om.eltwise({eltwise1,relu10}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res2c/Add#249");

    std::vector<int64_t> weightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch1#48_weights#49.dat");
    auto weights11 = om.constantInt(weightsData11,{1,1,256,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3a_branch1#48_weights#49");
    auto conv11 = om.conv(eltwise2, weights11, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch1#318");

    std::vector<int64_t> biasWeightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch1#48_bias#50.dat");
    auto biasWeights11 = om.constantInt(biasWeightsData11,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3a_branch1#48_bias#50");
    auto bias_c11 = om.bias(conv11, biasWeights11, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu11 = om.relu(bias_c11, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch1#250");

    std::vector<int64_t> weightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2a#52_weights#53.dat");
    auto weights12 = om.constantInt(weightsData12,{1,1,256,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3a_branch2a#52_weights#53");
    auto conv12 = om.conv(eltwise2, weights12, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2a#319");

    std::vector<int64_t> biasWeightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2a#52_bias#54.dat");
    auto biasWeights12 = om.constantInt(biasWeightsData12,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3a_branch2a#52_bias#54");
    auto bias_c12 = om.bias(conv12, biasWeights12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu12 = om.relu(bias_c12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2a#251");

    std::vector<int64_t> weightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2b#56_weights#57.dat");
    auto weights13 = om.constantInt(weightsData13,{3,3,128,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3a_branch2b#56_weights#57");
    auto conv13 = om.conv(relu12, weights13, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2b#320");

    std::vector<int64_t> biasWeightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2b#56_bias#58.dat");
    auto biasWeights13 = om.constantInt(biasWeightsData13,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3a_branch2b#56_bias#58");
    auto bias_c13 = om.bias(conv13, biasWeights13, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu13 = om.relu(bias_c13, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2b#252");

    std::vector<int64_t> weightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2c#60_weights#61.dat");
    auto weights14 = om.constantInt(weightsData14,{1,1,128,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3a_branch2c#60_weights#61");
    auto conv14 = om.conv(relu13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2c#321");

    std::vector<int64_t> biasWeightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3a_branch2c#60_bias#62.dat");
    auto biasWeights14 = om.constantInt(biasWeightsData14,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3a_branch2c#60_bias#62");
    auto bias_c14 = om.bias(conv14, biasWeights14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu14 = om.relu(bias_c14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a_branch2c#253");

    auto eltwise3 = om.eltwise({relu11,relu14}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3a/Add#254");

    std::vector<int64_t> weightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2a#65_weights#66.dat");
    auto weights15 = om.constantInt(weightsData15,{1,1,512,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3b_branch2a#65_weights#66");
    auto conv15 = om.conv(eltwise3, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2a#322");

    std::vector<int64_t> biasWeightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2a#65_bias#67.dat");
    auto biasWeights15 = om.constantInt(biasWeightsData15,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3b_branch2a#65_bias#67");
    auto bias_c15 = om.bias(conv15, biasWeights15, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu15 = om.relu(bias_c15, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2a#255");

    std::vector<int64_t> weightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2b#69_weights#70.dat");
    auto weights16 = om.constantInt(weightsData16,{3,3,128,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3b_branch2b#69_weights#70");
    auto conv16 = om.conv(relu15, weights16, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2b#323");

    std::vector<int64_t> biasWeightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2b#69_bias#71.dat");
    auto biasWeights16 = om.constantInt(biasWeightsData16,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3b_branch2b#69_bias#71");
    auto bias_c16 = om.bias(conv16, biasWeights16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu16 = om.relu(bias_c16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2b#256");

    std::vector<int64_t> weightsData17 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2c#73_weights#74.dat");
    auto weights17 = om.constantInt(weightsData17,{1,1,128,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3b_branch2c#73_weights#74");
    auto conv17 = om.conv(relu16, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2c#324");

    std::vector<int64_t> biasWeightsData17 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3b_branch2c#73_bias#75.dat");
    auto biasWeights17 = om.constantInt(biasWeightsData17,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3b_branch2c#73_bias#75");
    auto bias_c17 = om.bias(conv17, biasWeights17, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu17 = om.relu(bias_c17, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b_branch2c#257");

    auto eltwise4 = om.eltwise({eltwise3,relu17}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3b/Add#258");

    std::vector<int64_t> weightsData18 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2a#78_weights#79.dat");
    auto weights18 = om.constantInt(weightsData18,{1,1,512,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3c_branch2a#78_weights#79");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2a#325");

    std::vector<int64_t> biasWeightsData18 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2a#78_bias#80.dat");
    auto biasWeights18 = om.constantInt(biasWeightsData18,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3c_branch2a#78_bias#80");
    auto bias_c18 = om.bias(conv18, biasWeights18, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu18 = om.relu(bias_c18, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2a#259");

    std::vector<int64_t> weightsData19 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2b#82_weights#83.dat");
    auto weights19 = om.constantInt(weightsData19,{3,3,128,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3c_branch2b#82_weights#83");
    auto conv19 = om.conv(relu18, weights19, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2b#326");

    std::vector<int64_t> biasWeightsData19 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2b#82_bias#84.dat");
    auto biasWeights19 = om.constantInt(biasWeightsData19,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3c_branch2b#82_bias#84");
    auto bias_c19 = om.bias(conv19, biasWeights19, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu19 = om.relu(bias_c19, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2b#260");

    std::vector<int64_t> weightsData20 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2c#86_weights#87.dat");
    auto weights20 = om.constantInt(weightsData20,{1,1,128,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3c_branch2c#86_weights#87");
    auto conv20 = om.conv(relu19, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2c#327");

    std::vector<int64_t> biasWeightsData20 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3c_branch2c#86_bias#88.dat");
    auto biasWeights20 = om.constantInt(biasWeightsData20,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3c_branch2c#86_bias#88");
    auto bias_c20 = om.bias(conv20, biasWeights20, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu20 = om.relu(bias_c20, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c_branch2c#261");

    auto eltwise5 = om.eltwise({eltwise4,relu20}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3c/Add#262");

    std::vector<int64_t> weightsData21 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2a#91_weights#92.dat");
    auto weights21 = om.constantInt(weightsData21,{1,1,512,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3d_branch2a#91_weights#92");
    auto conv21 = om.conv(eltwise5, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2a#328");

    std::vector<int64_t> biasWeightsData21 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2a#91_bias#93.dat");
    auto biasWeights21 = om.constantInt(biasWeightsData21,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3d_branch2a#91_bias#93");
    auto bias_c21 = om.bias(conv21, biasWeights21, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu21 = om.relu(bias_c21, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2a#263");

    std::vector<int64_t> weightsData22 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2b#95_weights#96.dat");
    auto weights22 = om.constantInt(weightsData22,{3,3,128,128}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3d_branch2b#95_weights#96");
    auto conv22 = om.conv(relu21, weights22, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2b#329");

    std::vector<int64_t> biasWeightsData22 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2b#95_bias#97.dat");
    auto biasWeights22 = om.constantInt(biasWeightsData22,{128}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3d_branch2b#95_bias#97");
    auto bias_c22 = om.bias(conv22, biasWeights22, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu22 = om.relu(bias_c22, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2b#264");

    std::vector<int64_t> weightsData23 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2c#99_weights#100.dat");
    auto weights23 = om.constantInt(weightsData23,{1,1,128,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res3d_branch2c#99_weights#100");
    auto conv23 = om.conv(relu22, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2c#330");

    std::vector<int64_t> biasWeightsData23 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res3d_branch2c#99_bias#101.dat");
    auto biasWeights23 = om.constantInt(biasWeightsData23,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res3d_branch2c#99_bias#101");
    auto bias_c23 = om.bias(conv23, biasWeights23, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu23 = om.relu(bias_c23, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d_branch2c#265");

    auto eltwise6 = om.eltwise({eltwise5,relu23}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res3d/Add#266");

    std::vector<int64_t> weightsData24 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch1#104_weights#105.dat");
    auto weights24 = om.constantInt(weightsData24,{1,1,512,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4a_branch1#104_weights#105");
    auto conv24 = om.conv(eltwise6, weights24, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch1#331");

    std::vector<int64_t> biasWeightsData24 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch1#104_bias#106.dat");
    auto biasWeights24 = om.constantInt(biasWeightsData24,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4a_branch1#104_bias#106");
    auto bias_c24 = om.bias(conv24, biasWeights24, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu24 = om.relu(bias_c24, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch1#267");

    std::vector<int64_t> weightsData25 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2a#108_weights#109.dat");
    auto weights25 = om.constantInt(weightsData25,{1,1,512,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4a_branch2a#108_weights#109");
    auto conv25 = om.conv(eltwise6, weights25, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2a#332");

    std::vector<int64_t> biasWeightsData25 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2a#108_bias#110.dat");
    auto biasWeights25 = om.constantInt(biasWeightsData25,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4a_branch2a#108_bias#110");
    auto bias_c25 = om.bias(conv25, biasWeights25, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu25 = om.relu(bias_c25, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2a#268");

    std::vector<int64_t> weightsData26 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2b#112_weights#113.dat");
    auto weights26 = om.constantInt(weightsData26,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4a_branch2b#112_weights#113");
    auto conv26 = om.conv(relu25, weights26, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2b#333");

    std::vector<int64_t> biasWeightsData26 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2b#112_bias#114.dat");
    auto biasWeights26 = om.constantInt(biasWeightsData26,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4a_branch2b#112_bias#114");
    auto bias_c26 = om.bias(conv26, biasWeights26, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu26 = om.relu(bias_c26, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2b#269");

    std::vector<int64_t> weightsData27 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2c#116_weights#117.dat");
    auto weights27 = om.constantInt(weightsData27,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4a_branch2c#116_weights#117");
    auto conv27 = om.conv(relu26, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2c#334");

    std::vector<int64_t> biasWeightsData27 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4a_branch2c#116_bias#118.dat");
    auto biasWeights27 = om.constantInt(biasWeightsData27,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4a_branch2c#116_bias#118");
    auto bias_c27 = om.bias(conv27, biasWeights27, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu27 = om.relu(bias_c27, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a_branch2c#270");

    auto eltwise7 = om.eltwise({relu24,relu27}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4a/Add#271");

    std::vector<int64_t> weightsData28 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2a#121_weights#122.dat");
    auto weights28 = om.constantInt(weightsData28,{1,1,1024,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4b_branch2a#121_weights#122");
    auto conv28 = om.conv(eltwise7, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2a#335");

    std::vector<int64_t> biasWeightsData28 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2a#121_bias#123.dat");
    auto biasWeights28 = om.constantInt(biasWeightsData28,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4b_branch2a#121_bias#123");
    auto bias_c28 = om.bias(conv28, biasWeights28, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu28 = om.relu(bias_c28, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2a#272");

    std::vector<int64_t> weightsData29 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2b#125_weights#126.dat");
    auto weights29 = om.constantInt(weightsData29,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4b_branch2b#125_weights#126");
    auto conv29 = om.conv(relu28, weights29, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2b#336");

    std::vector<int64_t> biasWeightsData29 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2b#125_bias#127.dat");
    auto biasWeights29 = om.constantInt(biasWeightsData29,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4b_branch2b#125_bias#127");
    auto bias_c29 = om.bias(conv29, biasWeights29, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu29 = om.relu(bias_c29, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2b#273");

    std::vector<int64_t> weightsData30 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2c#129_weights#130.dat");
    auto weights30 = om.constantInt(weightsData30,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4b_branch2c#129_weights#130");
    auto conv30 = om.conv(relu29, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2c#337");

    std::vector<int64_t> biasWeightsData30 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4b_branch2c#129_bias#131.dat");
    auto biasWeights30 = om.constantInt(biasWeightsData30,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4b_branch2c#129_bias#131");
    auto bias_c30 = om.bias(conv30, biasWeights30, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu30 = om.relu(bias_c30, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b_branch2c#274");

    auto eltwise8 = om.eltwise({eltwise7,relu30}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4b/Add#275");

    std::vector<int64_t> weightsData31 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2a#134_weights#135.dat");
    auto weights31 = om.constantInt(weightsData31,{1,1,1024,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4c_branch2a#134_weights#135");
    auto conv31 = om.conv(eltwise8, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2a#338");

    std::vector<int64_t> biasWeightsData31 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2a#134_bias#136.dat");
    auto biasWeights31 = om.constantInt(biasWeightsData31,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4c_branch2a#134_bias#136");
    auto bias_c31 = om.bias(conv31, biasWeights31, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu31 = om.relu(bias_c31, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2a#276");

    std::vector<int64_t> weightsData32 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2b#138_weights#139.dat");
    auto weights32 = om.constantInt(weightsData32,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4c_branch2b#138_weights#139");
    auto conv32 = om.conv(relu31, weights32, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2b#339");

    std::vector<int64_t> biasWeightsData32 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2b#138_bias#140.dat");
    auto biasWeights32 = om.constantInt(biasWeightsData32,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4c_branch2b#138_bias#140");
    auto bias_c32 = om.bias(conv32, biasWeights32, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu32 = om.relu(bias_c32, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2b#277");

    std::vector<int64_t> weightsData33 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2c#142_weights#143.dat");
    auto weights33 = om.constantInt(weightsData33,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4c_branch2c#142_weights#143");
    auto conv33 = om.conv(relu32, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2c#340");

    std::vector<int64_t> biasWeightsData33 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4c_branch2c#142_bias#144.dat");
    auto biasWeights33 = om.constantInt(biasWeightsData33,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4c_branch2c#142_bias#144");
    auto bias_c33 = om.bias(conv33, biasWeights33, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu33 = om.relu(bias_c33, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c_branch2c#278");

    auto eltwise9 = om.eltwise({eltwise8,relu33}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4c/Add#279");

    std::vector<int64_t> weightsData34 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2a#147_weights#148.dat");
    auto weights34 = om.constantInt(weightsData34,{1,1,1024,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4d_branch2a#147_weights#148");
    auto conv34 = om.conv(eltwise9, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2a#341");

    std::vector<int64_t> biasWeightsData34 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2a#147_bias#149.dat");
    auto biasWeights34 = om.constantInt(biasWeightsData34,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4d_branch2a#147_bias#149");
    auto bias_c34 = om.bias(conv34, biasWeights34, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu34 = om.relu(bias_c34, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2a#280");

    std::vector<int64_t> weightsData35 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2b#151_weights#152.dat");
    auto weights35 = om.constantInt(weightsData35,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4d_branch2b#151_weights#152");
    auto conv35 = om.conv(relu34, weights35, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2b#342");

    std::vector<int64_t> biasWeightsData35 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2b#151_bias#153.dat");
    auto biasWeights35 = om.constantInt(biasWeightsData35,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4d_branch2b#151_bias#153");
    auto bias_c35 = om.bias(conv35, biasWeights35, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu35 = om.relu(bias_c35, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2b#281");

    std::vector<int64_t> weightsData36 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2c#155_weights#156.dat");
    auto weights36 = om.constantInt(weightsData36,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4d_branch2c#155_weights#156");
    auto conv36 = om.conv(relu35, weights36, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2c#343");

    std::vector<int64_t> biasWeightsData36 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4d_branch2c#155_bias#157.dat");
    auto biasWeights36 = om.constantInt(biasWeightsData36,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4d_branch2c#155_bias#157");
    auto bias_c36 = om.bias(conv36, biasWeights36, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu36 = om.relu(bias_c36, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d_branch2c#282");

    auto eltwise10 = om.eltwise({eltwise9,relu36}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4d/Add#283");

    std::vector<int64_t> weightsData37 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2a#160_weights#161.dat");
    auto weights37 = om.constantInt(weightsData37,{1,1,1024,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4e_branch2a#160_weights#161");
    auto conv37 = om.conv(eltwise10, weights37, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2a#344");

    std::vector<int64_t> biasWeightsData37 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2a#160_bias#162.dat");
    auto biasWeights37 = om.constantInt(biasWeightsData37,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4e_branch2a#160_bias#162");
    auto bias_c37 = om.bias(conv37, biasWeights37, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu37 = om.relu(bias_c37, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2a#284");

    std::vector<int64_t> weightsData38 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2b#164_weights#165.dat");
    auto weights38 = om.constantInt(weightsData38,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4e_branch2b#164_weights#165");
    auto conv38 = om.conv(relu37, weights38, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2b#345");

    std::vector<int64_t> biasWeightsData38 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2b#164_bias#166.dat");
    auto biasWeights38 = om.constantInt(biasWeightsData38,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4e_branch2b#164_bias#166");
    auto bias_c38 = om.bias(conv38, biasWeights38, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu38 = om.relu(bias_c38, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2b#285");

    std::vector<int64_t> weightsData39 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2c#168_weights#169.dat");
    auto weights39 = om.constantInt(weightsData39,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4e_branch2c#168_weights#169");
    auto conv39 = om.conv(relu38, weights39, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2c#346");

    std::vector<int64_t> biasWeightsData39 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4e_branch2c#168_bias#170.dat");
    auto biasWeights39 = om.constantInt(biasWeightsData39,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4e_branch2c#168_bias#170");
    auto bias_c39 = om.bias(conv39, biasWeights39, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu39 = om.relu(bias_c39, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e_branch2c#286");

    auto eltwise11 = om.eltwise({eltwise10,relu39}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4e/Add#287");

    std::vector<int64_t> weightsData40 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2a#173_weights#174.dat");
    auto weights40 = om.constantInt(weightsData40,{1,1,1024,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4f_branch2a#173_weights#174");
    auto conv40 = om.conv(eltwise11, weights40, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2a#347");

    std::vector<int64_t> biasWeightsData40 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2a#173_bias#175.dat");
    auto biasWeights40 = om.constantInt(biasWeightsData40,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4f_branch2a#173_bias#175");
    auto bias_c40 = om.bias(conv40, biasWeights40, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu40 = om.relu(bias_c40, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2a#288");

    std::vector<int64_t> weightsData41 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2b#177_weights#178.dat");
    auto weights41 = om.constantInt(weightsData41,{3,3,256,256}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4f_branch2b#177_weights#178");
    auto conv41 = om.conv(relu40, weights41, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2b#348");

    std::vector<int64_t> biasWeightsData41 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2b#177_bias#179.dat");
    auto biasWeights41 = om.constantInt(biasWeightsData41,{256}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4f_branch2b#177_bias#179");
    auto bias_c41 = om.bias(conv41, biasWeights41, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu41 = om.relu(bias_c41, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2b#289");

    std::vector<int64_t> weightsData42 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2c#181_weights#182.dat");
    auto weights42 = om.constantInt(weightsData42,{1,1,256,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res4f_branch2c#181_weights#182");
    auto conv42 = om.conv(relu41, weights42, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2c#349");

    std::vector<int64_t> biasWeightsData42 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res4f_branch2c#181_bias#183.dat");
    auto biasWeights42 = om.constantInt(biasWeightsData42,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res4f_branch2c#181_bias#183");
    auto bias_c42 = om.bias(conv42, biasWeights42, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu42 = om.relu(bias_c42, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f_branch2c#290");

    auto eltwise12 = om.eltwise({eltwise11,relu42}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res4f/Add#291");

    std::vector<int64_t> weightsData43 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch1#186_weights#187.dat");
    auto weights43 = om.constantInt(weightsData43,{1,1,1024,2048}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5a_branch1#186_weights#187");
    auto conv43 = om.conv(eltwise12, weights43, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch1#350");

    std::vector<int64_t> biasWeightsData43 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch1#186_bias#188.dat");
    auto biasWeights43 = om.constantInt(biasWeightsData43,{2048}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5a_branch1#186_bias#188");
    auto bias_c43 = om.bias(conv43, biasWeights43, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu43 = om.relu(bias_c43, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch1#292");

    std::vector<int64_t> weightsData44 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2a#190_weights#191.dat");
    auto weights44 = om.constantInt(weightsData44,{1,1,1024,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5a_branch2a#190_weights#191");
    auto conv44 = om.conv(eltwise12, weights44, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2a#351");

    std::vector<int64_t> biasWeightsData44 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2a#190_bias#192.dat");
    auto biasWeights44 = om.constantInt(biasWeightsData44,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5a_branch2a#190_bias#192");
    auto bias_c44 = om.bias(conv44, biasWeights44, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu44 = om.relu(bias_c44, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2a#293");

    std::vector<int64_t> weightsData45 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2b#194_weights#195.dat");
    auto weights45 = om.constantInt(weightsData45,{3,3,512,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5a_branch2b#194_weights#195");
    auto conv45 = om.conv(relu44, weights45, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2b#352");

    std::vector<int64_t> biasWeightsData45 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2b#194_bias#196.dat");
    auto biasWeights45 = om.constantInt(biasWeightsData45,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5a_branch2b#194_bias#196");
    auto bias_c45 = om.bias(conv45, biasWeights45, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu45 = om.relu(bias_c45, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2b#294");

    std::vector<int64_t> weightsData46 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2c#198_weights#199.dat");
    auto weights46 = om.constantInt(weightsData46,{1,1,512,2048}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5a_branch2c#198_weights#199");
    auto conv46 = om.conv(relu45, weights46, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2c#353");

    std::vector<int64_t> biasWeightsData46 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5a_branch2c#198_bias#200.dat");
    auto biasWeights46 = om.constantInt(biasWeightsData46,{2048}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5a_branch2c#198_bias#200");
    auto bias_c46 = om.bias(conv46, biasWeights46, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu46 = om.relu(bias_c46, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a_branch2c#295");

    auto eltwise13 = om.eltwise({relu43,relu46}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5a/Add#296");

    std::vector<int64_t> weightsData47 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2a#203_weights#204.dat");
    auto weights47 = om.constantInt(weightsData47,{1,1,2048,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5b_branch2a#203_weights#204");
    auto conv47 = om.conv(eltwise13, weights47, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2a#354");

    std::vector<int64_t> biasWeightsData47 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2a#203_bias#205.dat");
    auto biasWeights47 = om.constantInt(biasWeightsData47,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5b_branch2a#203_bias#205");
    auto bias_c47 = om.bias(conv47, biasWeights47, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu47 = om.relu(bias_c47, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2a#297");

    std::vector<int64_t> weightsData48 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2b#207_weights#208.dat");
    auto weights48 = om.constantInt(weightsData48,{3,3,512,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5b_branch2b#207_weights#208");
    auto conv48 = om.conv(relu47, weights48, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2b#355");

    std::vector<int64_t> biasWeightsData48 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2b#207_bias#209.dat");
    auto biasWeights48 = om.constantInt(biasWeightsData48,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5b_branch2b#207_bias#209");
    auto bias_c48 = om.bias(conv48, biasWeights48, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu48 = om.relu(bias_c48, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2b#298");

    std::vector<int64_t> weightsData49 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2c#211_weights#212.dat");
    auto weights49 = om.constantInt(weightsData49,{1,1,512,2048}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5b_branch2c#211_weights#212");
    auto conv49 = om.conv(relu48, weights49, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2c#356");

    std::vector<int64_t> biasWeightsData49 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5b_branch2c#211_bias#213.dat");
    auto biasWeights49 = om.constantInt(biasWeightsData49,{2048}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5b_branch2c#211_bias#213");
    auto bias_c49 = om.bias(conv49, biasWeights49, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu49 = om.relu(bias_c49, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b_branch2c#299");

    auto eltwise14 = om.eltwise({eltwise13,relu49}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5b/Add#300");

    std::vector<int64_t> weightsData50 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2a#216_weights#217.dat");
    auto weights50 = om.constantInt(weightsData50,{1,1,2048,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5c_branch2a#216_weights#217");
    auto conv50 = om.conv(eltwise14, weights50, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2a#357");

    std::vector<int64_t> biasWeightsData50 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2a#216_bias#218.dat");
    auto biasWeights50 = om.constantInt(biasWeightsData50,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5c_branch2a#216_bias#218");
    auto bias_c50 = om.bias(conv50, biasWeights50, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu50 = om.relu(bias_c50, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2a#301");

    std::vector<int64_t> weightsData51 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2b#220_weights#221.dat");
    auto weights51 = om.constantInt(weightsData51,{3,3,512,512}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5c_branch2b#220_weights#221");
    auto conv51 = om.conv(relu50, weights51, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2b#358");

    std::vector<int64_t> biasWeightsData51 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2b#220_bias#222.dat");
    auto biasWeights51 = om.constantInt(biasWeightsData51,{512}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5c_branch2b#220_bias#222");
    auto bias_c51 = om.bias(conv51, biasWeights51, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu51 = om.relu(bias_c51, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2b#302");

    std::vector<int64_t> weightsData52 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2c#224_weights#225.dat");
    auto weights52 = om.constantInt(weightsData52,{1,1,512,2048}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "res5c_branch2c#224_weights#225");
    auto conv52 = om.conv(relu51, weights52, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2c#359");

    std::vector<int64_t> biasWeightsData52 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/res5c_branch2c#224_bias#226.dat");
    auto biasWeights52 = om.constantInt(biasWeightsData52,{2048}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "res5c_branch2c#224_bias#226");
    auto bias_c52 = om.bias(conv52, biasWeights52, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu52 = om.relu(bias_c52, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c_branch2c#303");

    auto eltwise15 = om.eltwise({eltwise14,relu52}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "res5c/Add#304");

    auto pool1 = om.averagePool(eltwise15, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "pool5/AvgPool#305");

    std::vector<int64_t> weightsData53 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/fc1000#230_weights#231.dat");
    auto weights53 = om.constantInt(weightsData53,{1,1,2048,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "fc1000#230_weights#231");
    auto conv53 = om.conv(pool1, weights53, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "fc1000#360");

    std::vector<int64_t> biasWeightsData53 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/fc1000#230_bias#232.dat");
    auto biasWeights53 = om.constantInt(biasWeightsData53,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "fc1000#230_bias#232");
    auto bias_c53 = om.bias(conv53, biasWeights53, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu53 = om.relu(bias_c53, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "fc1000#306");

    om.output(relu53);

    return compilationUnit;
}
