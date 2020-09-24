#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

#include "mobilenetv2_fp16.hpp"

double inf = std::numeric_limits<double>::infinity();

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

mv::CompilationUnit buildMobilenetV2_fp16(const std::string& binaryDir)
{
    using std::int32_t;
    using std::int64_t;
    using std::uint8_t;

    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit compilationUnit("mobilenetv2_fp16");
    mv::OpModel& om = compilationUnit.model();

    auto input0 = om.input({224,224,3,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input#205");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Conv_Relu6#0_weights#1.dat");
    auto weights0 = om.constantInt(weightsData0,{3,3,3,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "Conv/Relu6#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Conv/Relu6#270");

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Conv_Relu6#0_bias#2.dat");
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "Conv/Relu6#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu0 = om.relu(bias_c0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Conv/Relu6#206");

    std::vector<int64_t> d_weightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_depthwise_Relu6#4_weights#5.dat");
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,32,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv/depthwise/Relu6#4_weights#5");
    auto depthConv0 = om.depthwiseConv(relu0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv/depthwise/Relu6#271");

    std::vector<int64_t> biasd_WeightsData0 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_depthwise_Relu6#4_bias#6.dat");
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv/depthwise/Relu6#4_bias#6");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu1 = om.relu(bias_cd0, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv/depthwise/Relu6#207");

    std::vector<int64_t> weightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_project_BatchNorm_FusedBatchNorm_BiasAdd#8_weights#9.dat");
    auto weights1 = om.constantInt(weightsData1,{1,1,32,16}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#8_weights#9");
    auto conv1 = om.conv(relu1, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#208");

    std::vector<int64_t> biasWeightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_project_BatchNorm_FusedBatchNorm_BiasAdd#8_bias#10.dat");
    auto biasWeights1 = om.constantInt(biasWeightsData1,{16}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#8_bias#10");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_expand_Relu6#11_weights#12.dat");
    auto weights2 = om.constantInt(weightsData2,{1,1,16,96}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/expand/Relu6#11_weights#12");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/expand/Relu6#272");

    std::vector<int64_t> biasWeightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_expand_Relu6#11_bias#13.dat");
    auto biasWeights2 = om.constantInt(biasWeightsData2,{96}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/expand/Relu6#11_bias#13");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu2 = om.relu(bias_c2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/expand/Relu6#209");

    std::vector<int64_t> d_weightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_depthwise_Relu6#15_weights#16.dat");
    auto d_weights1 = om.constantInt(d_weightsData1,{3,3,96,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6#15_weights#16");
    auto depthConv1 = om.depthwiseConv(relu2, d_weights1, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6#273");

    std::vector<int64_t> biasd_WeightsData1 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_depthwise_Relu6#15_bias#17.dat");
    auto biasdWeights1 = om.constantInt(biasd_WeightsData1,{96}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6#15_bias#17");
    auto bias_cd1 = om.bias(depthConv1, biasdWeights1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu3 = om.relu(bias_cd1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6#210");

    std::vector<int64_t> weightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_project_BatchNorm_FusedBatchNorm_BiasAdd#19_weights#20.dat");
    auto weights3 = om.constantInt(weightsData3,{1,1,96,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#19_weights#20");
    auto conv3 = om.conv(relu3, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#211");

    std::vector<int64_t> biasWeightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_1_project_BatchNorm_FusedBatchNorm_BiasAdd#19_bias#21.dat");
    auto biasWeights3 = om.constantInt(biasWeightsData3,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#19_bias#21");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_depthwise_Relu6#22_weights#23.dat");
    auto weights4 = om.constantInt(weightsData4,{1,1,32,192}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6#22_weights#23");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6#274");

    std::vector<int64_t> biasWeightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_depthwise_Relu6#22_bias#24.dat");
    auto biasWeights4 = om.constantInt(biasWeightsData4,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6#22_bias#24");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu4 = om.relu(bias_c4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6#212");

    std::vector<int64_t> d_weightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_depthwise_Relu6_1#26_weights#27.dat");
    auto d_weights2 = om.constantInt(d_weightsData2,{3,3,192,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1#26_weights#27");
    auto depthConv2 = om.depthwiseConv(relu4, d_weights2, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1#275");

    std::vector<int64_t> biasd_WeightsData2 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_depthwise_Relu6_1#26_bias#28.dat");
    auto biasdWeights2 = om.constantInt(biasd_WeightsData2,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1#26_bias#28");
    auto bias_cd2 = om.bias(depthConv2, biasdWeights2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu5 = om.relu(bias_cd2, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1#213");

    std::vector<int64_t> weightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_project_BatchNorm_FusedBatchNorm_BiasAdd#30_weights#31.dat");
    auto weights5 = om.constantInt(weightsData5,{1,1,192,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#30_weights#31");
    auto conv5 = om.conv(relu5, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#214");

    std::vector<int64_t> biasWeightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_2_project_BatchNorm_FusedBatchNorm_BiasAdd#30_bias#32.dat");
    auto biasWeights5 = om.constantInt(biasWeightsData5,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#30_bias#32");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise0 = om.eltwise({bias_c3,bias_c5}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_2/add/Add#215");

    std::vector<int64_t> weightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_expand_Relu6#34_weights#35.dat");
    auto weights6 = om.constantInt(weightsData6,{1,1,32,192}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/expand/Relu6#34_weights#35");
    auto conv6 = om.conv(eltwise0, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/expand/Relu6#276");

    std::vector<int64_t> biasWeightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_expand_Relu6#34_bias#36.dat");
    auto biasWeights6 = om.constantInt(biasWeightsData6,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/expand/Relu6#34_bias#36");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu6 = om.relu(bias_c6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/expand/Relu6#216");

    std::vector<int64_t> d_weightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_depthwise_Relu6#38_weights#39.dat");
    auto d_weights3 = om.constantInt(d_weightsData3,{3,3,192,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6#38_weights#39");
    auto depthConv3 = om.depthwiseConv(relu6, d_weights3, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6#277");

    std::vector<int64_t> biasd_WeightsData3 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_depthwise_Relu6#38_bias#40.dat");
    auto biasdWeights3 = om.constantInt(biasd_WeightsData3,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6#38_bias#40");
    auto bias_cd3 = om.bias(depthConv3, biasdWeights3, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu7 = om.relu(bias_cd3, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6#217");

    std::vector<int64_t> weightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_project_BatchNorm_FusedBatchNorm_BiasAdd#42_weights#43.dat");
    auto weights7 = om.constantInt(weightsData7,{1,1,192,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#42_weights#43");
    auto conv7 = om.conv(relu7, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#218");

    std::vector<int64_t> biasWeightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_3_project_BatchNorm_FusedBatchNorm_BiasAdd#42_bias#44.dat");
    auto biasWeights7 = om.constantInt(biasWeightsData7,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#42_bias#44");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_expand_Relu6#45_weights#46.dat");
    auto weights8 = om.constantInt(weightsData8,{1,1,32,192}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/expand/Relu6#45_weights#46");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/expand/Relu6#278");

    std::vector<int64_t> biasWeightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_expand_Relu6#45_bias#47.dat");
    auto biasWeights8 = om.constantInt(biasWeightsData8,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/expand/Relu6#45_bias#47");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu8 = om.relu(bias_c8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/expand/Relu6#219");

    std::vector<int64_t> d_weightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_depthwise_Relu6#49_weights#50.dat");
    auto d_weights4 = om.constantInt(d_weightsData4,{3,3,192,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6#49_weights#50");
    auto depthConv4 = om.depthwiseConv(relu8, d_weights4, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6#279");

    std::vector<int64_t> biasd_WeightsData4 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_depthwise_Relu6#49_bias#51.dat");
    auto biasdWeights4 = om.constantInt(biasd_WeightsData4,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6#49_bias#51");
    auto bias_cd4 = om.bias(depthConv4, biasdWeights4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu9 = om.relu(bias_cd4, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6#220");

    std::vector<int64_t> weightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_project_BatchNorm_FusedBatchNorm_BiasAdd#53_weights#54.dat");
    auto weights9 = om.constantInt(weightsData9,{1,1,192,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#53_weights#54");
    auto conv9 = om.conv(relu9, weights9, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#221");

    std::vector<int64_t> biasWeightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_4_project_BatchNorm_FusedBatchNorm_BiasAdd#53_bias#55.dat");
    auto biasWeights9 = om.constantInt(biasWeightsData9,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#53_bias#55");
    auto bias_c9 = om.bias(conv9, biasWeights9, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise1 = om.eltwise({bias_c7,bias_c9}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_4/add/Add#222");

    std::vector<int64_t> weightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_expand_Relu6#57_weights#58.dat");
    auto weights10 = om.constantInt(weightsData10,{1,1,32,192}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/expand/Relu6#57_weights#58");
    auto conv10 = om.conv(eltwise1, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/expand/Relu6#280");

    std::vector<int64_t> biasWeightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_expand_Relu6#57_bias#59.dat");
    auto biasWeights10 = om.constantInt(biasWeightsData10,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/expand/Relu6#57_bias#59");
    auto bias_c10 = om.bias(conv10, biasWeights10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu10 = om.relu(bias_c10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/expand/Relu6#223");

    std::vector<int64_t> d_weightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_depthwise_Relu6#61_weights#62.dat");
    auto d_weights5 = om.constantInt(d_weightsData5,{3,3,192,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6#61_weights#62");
    auto depthConv5 = om.depthwiseConv(relu10, d_weights5, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6#281");

    std::vector<int64_t> biasd_WeightsData5 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_depthwise_Relu6#61_bias#63.dat");
    auto biasdWeights5 = om.constantInt(biasd_WeightsData5,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6#61_bias#63");
    auto bias_cd5 = om.bias(depthConv5, biasdWeights5, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu11 = om.relu(bias_cd5, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6#224");

    std::vector<int64_t> weightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_project_BatchNorm_FusedBatchNorm_BiasAdd#65_weights#66.dat");
    auto weights11 = om.constantInt(weightsData11,{1,1,192,32}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#65_weights#66");
    auto conv11 = om.conv(relu11, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#225");

    std::vector<int64_t> biasWeightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_5_project_BatchNorm_FusedBatchNorm_BiasAdd#65_bias#67.dat");
    auto biasWeights11 = om.constantInt(biasWeightsData11,{32}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#65_bias#67");
    auto bias_c11 = om.bias(conv11, biasWeights11, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise2 = om.eltwise({eltwise1,bias_c11}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_5/add/Add#226");

    std::vector<int64_t> weightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_expand_Relu6#69_weights#70.dat");
    auto weights12 = om.constantInt(weightsData12,{1,1,32,192}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/expand/Relu6#69_weights#70");
    auto conv12 = om.conv(eltwise2, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/expand/Relu6#282");

    std::vector<int64_t> biasWeightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_expand_Relu6#69_bias#71.dat");
    auto biasWeights12 = om.constantInt(biasWeightsData12,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/expand/Relu6#69_bias#71");
    auto bias_c12 = om.bias(conv12, biasWeights12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu12 = om.relu(bias_c12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/expand/Relu6#227");

    std::vector<int64_t> d_weightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_depthwise_Relu6#73_weights#74.dat");
    auto d_weights6 = om.constantInt(d_weightsData6,{3,3,192,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6#73_weights#74");
    auto depthConv6 = om.depthwiseConv(relu12, d_weights6, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6#283");

    std::vector<int64_t> biasd_WeightsData6 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_depthwise_Relu6#73_bias#75.dat");
    auto biasdWeights6 = om.constantInt(biasd_WeightsData6,{192}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6#73_bias#75");
    auto bias_cd6 = om.bias(depthConv6, biasdWeights6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu13 = om.relu(bias_cd6, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6#228");

    std::vector<int64_t> weightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_project_BatchNorm_FusedBatchNorm_BiasAdd#77_weights#78.dat");
    auto weights13 = om.constantInt(weightsData13,{1,1,192,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#77_weights#78");
    auto conv13 = om.conv(relu13, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#229");

    std::vector<int64_t> biasWeightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_6_project_BatchNorm_FusedBatchNorm_BiasAdd#77_bias#79.dat");
    auto biasWeights13 = om.constantInt(biasWeightsData13,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#77_bias#79");
    auto bias_c13 = om.bias(conv13, biasWeights13, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_expand_Relu6#80_weights#81.dat");
    auto weights14 = om.constantInt(weightsData14,{1,1,64,384}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/expand/Relu6#80_weights#81");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/expand/Relu6#284");

    std::vector<int64_t> biasWeightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_expand_Relu6#80_bias#82.dat");
    auto biasWeights14 = om.constantInt(biasWeightsData14,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/expand/Relu6#80_bias#82");
    auto bias_c14 = om.bias(conv14, biasWeights14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu14 = om.relu(bias_c14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/expand/Relu6#230");

    std::vector<int64_t> d_weightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_depthwise_Relu6#84_weights#85.dat");
    auto d_weights7 = om.constantInt(d_weightsData7,{3,3,384,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6#84_weights#85");
    auto depthConv7 = om.depthwiseConv(relu14, d_weights7, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6#285");

    std::vector<int64_t> biasd_WeightsData7 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_depthwise_Relu6#84_bias#86.dat");
    auto biasdWeights7 = om.constantInt(biasd_WeightsData7,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6#84_bias#86");
    auto bias_cd7 = om.bias(depthConv7, biasdWeights7, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu15 = om.relu(bias_cd7, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6#231");

    std::vector<int64_t> weightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_project_BatchNorm_FusedBatchNorm_BiasAdd#88_weights#89.dat");
    auto weights15 = om.constantInt(weightsData15,{1,1,384,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#88_weights#89");
    auto conv15 = om.conv(relu15, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#232");

    std::vector<int64_t> biasWeightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_7_project_BatchNorm_FusedBatchNorm_BiasAdd#88_bias#90.dat");
    auto biasWeights15 = om.constantInt(biasWeightsData15,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#88_bias#90");
    auto bias_c15 = om.bias(conv15, biasWeights15, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise3 = om.eltwise({bias_c13,bias_c15}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_7/add/Add#233");

    std::vector<int64_t> weightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_expand_Relu6#92_weights#93.dat");
    auto weights16 = om.constantInt(weightsData16,{1,1,64,384}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/expand/Relu6#92_weights#93");
    auto conv16 = om.conv(eltwise3, weights16, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/expand/Relu6#286");

    std::vector<int64_t> biasWeightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_expand_Relu6#92_bias#94.dat");
    auto biasWeights16 = om.constantInt(biasWeightsData16,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/expand/Relu6#92_bias#94");
    auto bias_c16 = om.bias(conv16, biasWeights16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu16 = om.relu(bias_c16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/expand/Relu6#234");

    std::vector<int64_t> d_weightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_depthwise_Relu6#96_weights#97.dat");
    auto d_weights8 = om.constantInt(d_weightsData8,{3,3,384,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6#96_weights#97");
    auto depthConv8 = om.depthwiseConv(relu16, d_weights8, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6#287");

    std::vector<int64_t> biasd_WeightsData8 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_depthwise_Relu6#96_bias#98.dat");
    auto biasdWeights8 = om.constantInt(biasd_WeightsData8,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6#96_bias#98");
    auto bias_cd8 = om.bias(depthConv8, biasdWeights8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu17 = om.relu(bias_cd8, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6#235");

    std::vector<int64_t> weightsData17 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_project_BatchNorm_FusedBatchNorm_BiasAdd#100_weights#101.dat");
    auto weights17 = om.constantInt(weightsData17,{1,1,384,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#100_weights#101");
    auto conv17 = om.conv(relu17, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#236");

    std::vector<int64_t> biasWeightsData17 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_8_project_BatchNorm_FusedBatchNorm_BiasAdd#100_bias#102.dat");
    auto biasWeights17 = om.constantInt(biasWeightsData17,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#100_bias#102");
    auto bias_c17 = om.bias(conv17, biasWeights17, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise4 = om.eltwise({eltwise3,bias_c17}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_8/add/Add#237");

    std::vector<int64_t> weightsData18 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_expand_Relu6#104_weights#105.dat");
    auto weights18 = om.constantInt(weightsData18,{1,1,64,384}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/expand/Relu6#104_weights#105");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/expand/Relu6#288");

    std::vector<int64_t> biasWeightsData18 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_expand_Relu6#104_bias#106.dat");
    auto biasWeights18 = om.constantInt(biasWeightsData18,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/expand/Relu6#104_bias#106");
    auto bias_c18 = om.bias(conv18, biasWeights18, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu18 = om.relu(bias_c18, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/expand/Relu6#238");

    std::vector<int64_t> d_weightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_depthwise_Relu6#108_weights#109.dat");
    auto d_weights9 = om.constantInt(d_weightsData9,{3,3,384,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6#108_weights#109");
    auto depthConv9 = om.depthwiseConv(relu18, d_weights9, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6#289");

    std::vector<int64_t> biasd_WeightsData9 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_depthwise_Relu6#108_bias#110.dat");
    auto biasdWeights9 = om.constantInt(biasd_WeightsData9,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6#108_bias#110");
    auto bias_cd9 = om.bias(depthConv9, biasdWeights9, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu19 = om.relu(bias_cd9, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6#239");

    std::vector<int64_t> weightsData19 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_project_BatchNorm_FusedBatchNorm_BiasAdd#112_weights#113.dat");
    auto weights19 = om.constantInt(weightsData19,{1,1,384,64}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#112_weights#113");
    auto conv19 = om.conv(relu19, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#240");

    std::vector<int64_t> biasWeightsData19 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_9_project_BatchNorm_FusedBatchNorm_BiasAdd#112_bias#114.dat");
    auto biasWeights19 = om.constantInt(biasWeightsData19,{64}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#112_bias#114");
    auto bias_c19 = om.bias(conv19, biasWeights19, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise5 = om.eltwise({eltwise4,bias_c19}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_9/add/Add#241");

    std::vector<int64_t> weightsData20 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_expand_Relu6#116_weights#117.dat");
    auto weights20 = om.constantInt(weightsData20,{1,1,64,384}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/expand/Relu6#116_weights#117");
    auto conv20 = om.conv(eltwise5, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/expand/Relu6#290");

    std::vector<int64_t> biasWeightsData20 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_expand_Relu6#116_bias#118.dat");
    auto biasWeights20 = om.constantInt(biasWeightsData20,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/expand/Relu6#116_bias#118");
    auto bias_c20 = om.bias(conv20, biasWeights20, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu20 = om.relu(bias_c20, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/expand/Relu6#242");

    std::vector<int64_t> d_weightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_depthwise_Relu6#120_weights#121.dat");
    auto d_weights10 = om.constantInt(d_weightsData10,{3,3,384,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6#120_weights#121");
    auto depthConv10 = om.depthwiseConv(relu20, d_weights10, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6#291");

    std::vector<int64_t> biasd_WeightsData10 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_depthwise_Relu6#120_bias#122.dat");
    auto biasdWeights10 = om.constantInt(biasd_WeightsData10,{384}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6#120_bias#122");
    auto bias_cd10 = om.bias(depthConv10, biasdWeights10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu21 = om.relu(bias_cd10, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6#243");

    std::vector<int64_t> weightsData21 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_project_BatchNorm_FusedBatchNorm_BiasAdd#124_weights#125.dat");
    auto weights21 = om.constantInt(weightsData21,{1,1,384,96}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#124_weights#125");
    auto conv21 = om.conv(relu21, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#244");

    std::vector<int64_t> biasWeightsData21 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_10_project_BatchNorm_FusedBatchNorm_BiasAdd#124_bias#126.dat");
    auto biasWeights21 = om.constantInt(biasWeightsData21,{96}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#124_bias#126");
    auto bias_c21 = om.bias(conv21, biasWeights21, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData22 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_expand_Relu6#127_weights#128.dat");
    auto weights22 = om.constantInt(weightsData22,{1,1,96,576}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/expand/Relu6#127_weights#128");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/expand/Relu6#292");

    std::vector<int64_t> biasWeightsData22 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_expand_Relu6#127_bias#129.dat");
    auto biasWeights22 = om.constantInt(biasWeightsData22,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/expand/Relu6#127_bias#129");
    auto bias_c22 = om.bias(conv22, biasWeights22, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu22 = om.relu(bias_c22, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/expand/Relu6#245");

    std::vector<int64_t> d_weightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_depthwise_Relu6#131_weights#132.dat");
    auto d_weights11 = om.constantInt(d_weightsData11,{3,3,576,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6#131_weights#132");
    auto depthConv11 = om.depthwiseConv(relu22, d_weights11, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6#293");

    std::vector<int64_t> biasd_WeightsData11 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_depthwise_Relu6#131_bias#133.dat");
    auto biasdWeights11 = om.constantInt(biasd_WeightsData11,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6#131_bias#133");
    auto bias_cd11 = om.bias(depthConv11, biasdWeights11, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu23 = om.relu(bias_cd11, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6#246");

    std::vector<int64_t> weightsData23 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_project_BatchNorm_FusedBatchNorm_BiasAdd#135_weights#136.dat");
    auto weights23 = om.constantInt(weightsData23,{1,1,576,96}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#135_weights#136");
    auto conv23 = om.conv(relu23, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#247");

    std::vector<int64_t> biasWeightsData23 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_11_project_BatchNorm_FusedBatchNorm_BiasAdd#135_bias#137.dat");
    auto biasWeights23 = om.constantInt(biasWeightsData23,{96}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#135_bias#137");
    auto bias_c23 = om.bias(conv23, biasWeights23, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise6 = om.eltwise({bias_c21,bias_c23}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_11/add/Add#248");

    std::vector<int64_t> weightsData24 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_expand_Relu6#139_weights#140.dat");
    auto weights24 = om.constantInt(weightsData24,{1,1,96,576}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/expand/Relu6#139_weights#140");
    auto conv24 = om.conv(eltwise6, weights24, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/expand/Relu6#294");

    std::vector<int64_t> biasWeightsData24 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_expand_Relu6#139_bias#141.dat");
    auto biasWeights24 = om.constantInt(biasWeightsData24,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/expand/Relu6#139_bias#141");
    auto bias_c24 = om.bias(conv24, biasWeights24, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu24 = om.relu(bias_c24, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/expand/Relu6#249");

    std::vector<int64_t> d_weightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_depthwise_Relu6#143_weights#144.dat");
    auto d_weights12 = om.constantInt(d_weightsData12,{3,3,576,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6#143_weights#144");
    auto depthConv12 = om.depthwiseConv(relu24, d_weights12, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6#295");

    std::vector<int64_t> biasd_WeightsData12 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_depthwise_Relu6#143_bias#145.dat");
    auto biasdWeights12 = om.constantInt(biasd_WeightsData12,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6#143_bias#145");
    auto bias_cd12 = om.bias(depthConv12, biasdWeights12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu25 = om.relu(bias_cd12, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6#250");

    std::vector<int64_t> weightsData25 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_project_BatchNorm_FusedBatchNorm_BiasAdd#147_weights#148.dat");
    auto weights25 = om.constantInt(weightsData25,{1,1,576,96}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#147_weights#148");
    auto conv25 = om.conv(relu25, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#251");

    std::vector<int64_t> biasWeightsData25 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_12_project_BatchNorm_FusedBatchNorm_BiasAdd#147_bias#149.dat");
    auto biasWeights25 = om.constantInt(biasWeightsData25,{96}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#147_bias#149");
    auto bias_c25 = om.bias(conv25, biasWeights25, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise7 = om.eltwise({eltwise6,bias_c25}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_12/add/Add#252");

    std::vector<int64_t> weightsData26 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_expand_Relu6#151_weights#152.dat");
    auto weights26 = om.constantInt(weightsData26,{1,1,96,576}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/expand/Relu6#151_weights#152");
    auto conv26 = om.conv(eltwise7, weights26, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/expand/Relu6#296");

    std::vector<int64_t> biasWeightsData26 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_expand_Relu6#151_bias#153.dat");
    auto biasWeights26 = om.constantInt(biasWeightsData26,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/expand/Relu6#151_bias#153");
    auto bias_c26 = om.bias(conv26, biasWeights26, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu26 = om.relu(bias_c26, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/expand/Relu6#253");

    std::vector<int64_t> d_weightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_depthwise_Relu6#155_weights#156.dat");
    auto d_weights13 = om.constantInt(d_weightsData13,{3,3,576,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6#155_weights#156");
    auto depthConv13 = om.depthwiseConv(relu26, d_weights13, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6#297");

    std::vector<int64_t> biasd_WeightsData13 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_depthwise_Relu6#155_bias#157.dat");
    auto biasdWeights13 = om.constantInt(biasd_WeightsData13,{576}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6#155_bias#157");
    auto bias_cd13 = om.bias(depthConv13, biasdWeights13, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu27 = om.relu(bias_cd13, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6#254");

    std::vector<int64_t> weightsData27 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_project_BatchNorm_FusedBatchNorm_BiasAdd#159_weights#160.dat");
    auto weights27 = om.constantInt(weightsData27,{1,1,576,160}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#159_weights#160");
    auto conv27 = om.conv(relu27, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#255");

    std::vector<int64_t> biasWeightsData27 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_13_project_BatchNorm_FusedBatchNorm_BiasAdd#159_bias#161.dat");
    auto biasWeights27 = om.constantInt(biasWeightsData27,{160}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#159_bias#161");
    auto bias_c27 = om.bias(conv27, biasWeights27, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData28 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_expand_Relu6#162_weights#163.dat");
    auto weights28 = om.constantInt(weightsData28,{1,1,160,960}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/expand/Relu6#162_weights#163");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/expand/Relu6#298");

    std::vector<int64_t> biasWeightsData28 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_expand_Relu6#162_bias#164.dat");
    auto biasWeights28 = om.constantInt(biasWeightsData28,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/expand/Relu6#162_bias#164");
    auto bias_c28 = om.bias(conv28, biasWeights28, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu28 = om.relu(bias_c28, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/expand/Relu6#256");

    std::vector<int64_t> d_weightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_depthwise_Relu6#166_weights#167.dat");
    auto d_weights14 = om.constantInt(d_weightsData14,{3,3,960,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6#166_weights#167");
    auto depthConv14 = om.depthwiseConv(relu28, d_weights14, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6#299");

    std::vector<int64_t> biasd_WeightsData14 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_depthwise_Relu6#166_bias#168.dat");
    auto biasdWeights14 = om.constantInt(biasd_WeightsData14,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6#166_bias#168");
    auto bias_cd14 = om.bias(depthConv14, biasdWeights14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu29 = om.relu(bias_cd14, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6#257");

    std::vector<int64_t> weightsData29 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_project_BatchNorm_FusedBatchNorm_BiasAdd#170_weights#171.dat");
    auto weights29 = om.constantInt(weightsData29,{1,1,960,160}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#170_weights#171");
    auto conv29 = om.conv(relu29, weights29, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#258");

    std::vector<int64_t> biasWeightsData29 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_14_project_BatchNorm_FusedBatchNorm_BiasAdd#170_bias#172.dat");
    auto biasWeights29 = om.constantInt(biasWeightsData29,{160}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#170_bias#172");
    auto bias_c29 = om.bias(conv29, biasWeights29, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise8 = om.eltwise({bias_c27,bias_c29}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_14/add/Add#259");

    std::vector<int64_t> weightsData30 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_expand_Relu6#174_weights#175.dat");
    auto weights30 = om.constantInt(weightsData30,{1,1,160,960}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/expand/Relu6#174_weights#175");
    auto conv30 = om.conv(eltwise8, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/expand/Relu6#300");

    std::vector<int64_t> biasWeightsData30 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_expand_Relu6#174_bias#176.dat");
    auto biasWeights30 = om.constantInt(biasWeightsData30,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/expand/Relu6#174_bias#176");
    auto bias_c30 = om.bias(conv30, biasWeights30, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu30 = om.relu(bias_c30, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/expand/Relu6#260");

    std::vector<int64_t> d_weightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_depthwise_Relu6#178_weights#179.dat");
    auto d_weights15 = om.constantInt(d_weightsData15,{3,3,960,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6#178_weights#179");
    auto depthConv15 = om.depthwiseConv(relu30, d_weights15, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6#301");

    std::vector<int64_t> biasd_WeightsData15 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_depthwise_Relu6#178_bias#180.dat");
    auto biasdWeights15 = om.constantInt(biasd_WeightsData15,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6#178_bias#180");
    auto bias_cd15 = om.bias(depthConv15, biasdWeights15, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu31 = om.relu(bias_cd15, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6#261");

    std::vector<int64_t> weightsData31 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_project_BatchNorm_FusedBatchNorm_BiasAdd#182_weights#183.dat");
    auto weights31 = om.constantInt(weightsData31,{1,1,960,160}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#182_weights#183");
    auto conv31 = om.conv(relu31, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#262");

    std::vector<int64_t> biasWeightsData31 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_15_project_BatchNorm_FusedBatchNorm_BiasAdd#182_bias#184.dat");
    auto biasWeights31 = om.constantInt(biasWeightsData31,{160}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#182_bias#184");
    auto bias_c31 = om.bias(conv31, biasWeights31, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto eltwise9 = om.eltwise({eltwise8,bias_c31}, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_15/add/Add#263");

    std::vector<int64_t> weightsData32 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_expand_Relu6#186_weights#187.dat");
    auto weights32 = om.constantInt(weightsData32,{1,1,160,960}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/expand/Relu6#186_weights#187");
    auto conv32 = om.conv(eltwise9, weights32, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/expand/Relu6#302");

    std::vector<int64_t> biasWeightsData32 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_expand_Relu6#186_bias#188.dat");
    auto biasWeights32 = om.constantInt(biasWeightsData32,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/expand/Relu6#186_bias#188");
    auto bias_c32 = om.bias(conv32, biasWeights32, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu32 = om.relu(bias_c32, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/expand/Relu6#264");

    std::vector<int64_t> d_weightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_depthwise_Relu6#190_weights#191.dat");
    auto d_weights16 = om.constantInt(d_weightsData16,{3,3,960,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6#190_weights#191");
    auto depthConv16 = om.depthwiseConv(relu32, d_weights16, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6#303");

    std::vector<int64_t> biasd_WeightsData16 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_depthwise_Relu6#190_bias#192.dat");
    auto biasdWeights16 = om.constantInt(biasd_WeightsData16,{960}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6#190_bias#192");
    auto bias_cd16 = om.bias(depthConv16, biasdWeights16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu33 = om.relu(bias_cd16, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6#265");

    std::vector<int64_t> weightsData33 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_project_BatchNorm_FusedBatchNorm_BiasAdd#194_weights#195.dat");
    auto weights33 = om.constantInt(weightsData33,{1,1,960,320}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#194_weights#195");
    auto conv33 = om.conv(relu33, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#266");

    std::vector<int64_t> biasWeightsData33 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/expanded_conv_16_project_BatchNorm_FusedBatchNorm_BiasAdd#194_bias#196.dat");
    auto biasWeights33 = om.constantInt(biasWeightsData33,{320}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#194_bias#196");
    auto bias_c33 = om.bias(conv33, biasWeights33, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    std::vector<int64_t> weightsData34 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Conv_1_Relu6#197_weights#198.dat");
    auto weights34 = om.constantInt(weightsData34,{1,1,320,1280}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "Conv_1/Relu6#197_weights#198");
    auto conv34 = om.conv(bias_c33, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Conv_1/Relu6#304");

    std::vector<int64_t> biasWeightsData34 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Conv_1_Relu6#197_bias#199.dat");
    auto biasWeights34 = om.constantInt(biasWeightsData34,{1280}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "Conv_1/Relu6#197_bias#199");
    auto bias_c34 = om.bias(conv34, biasWeights34, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu34 = om.relu(bias_c34, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Conv_1/Relu6#267");

    auto pool0 = om.averagePool(relu34, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Logits/AvgPool/AvgPool#268");

    std::vector<int64_t> weightsData35 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Logits_Conv2d_1c_1x1_BiasAdd_Reshape#202_weights#203.dat");
    auto weights35 = om.constantInt(weightsData35,{1,1,1280,1024}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "Logits/Conv2d_1c_1x1/BiasAdd/Reshape#202_weights#203");
    auto conv35 = om.conv(pool0, weights35, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "Logits/Conv2d_1c_1x1/BiasAdd/Reshape#269");

    std::vector<int64_t> biasWeightsData35 = read_weights_from_file<int64_t, uint16_t>(binaryDir + "weights_bias/Logits_Conv2d_1c_1x1_BiasAdd_Reshape#202_bias#204.dat");
    auto biasWeights35 = om.constantInt(biasWeightsData35,{1024}, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "Logits/Conv2d_1c_1x1/BiasAdd/Reshape#202_bias#204");
    auto bias_c35 = om.bias(conv35, biasWeights35, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    om.output(bias_c35);

    return compilationUnit;
}
