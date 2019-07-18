/**
 * @brief Example presenting composition and compilation of the ResNet50 CNN
 * 
 * In this example ResNet50 model is composed using MCMCompiler's Composition API. Then
 * the compilation for the target device MA2480 is initialized and compilation passes scheduled by 
 * the target descriptor are executed. Included GenerateDot pass will generate *.dot files
 * that visualize the computation model at the end of each accomplished compilation phase.
 * Included GenerateBlob pass will serialize the model to a binary deployable to the target device.
 * 
 * Notes:
 * - This implementation of ResNet50 uses fused batch norm representation - batch norm is expressed
 * as a sequence of scale and bias
 * - This implementation of ResNet50 is aligned with Caffe - batch norm is followed by scale and bias
 * - Weights and other model parameters are initialized as sequences of numbers starting with 0
 * 
 * @file cm_resnet50.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,2,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*2*32);
    auto weights0 = om.constantInt(weightsData0,{3,3,2,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "Conv/Relu6_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.076643613690976e-05},{-inf},{inf}}, "Conv/Relu6_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (3*3*32*1);
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,32,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{118},{0.0021302876994013786},{-0.2514924705028534},{0.2917308807373047}}, "expanded_conv/depthwise/Relu6_weights#4");
    auto depthConv0 = om.depthwiseConv(bias_c0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv/depthwise/Relu6#172");

    std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{8.354068995686248e-06},{-inf},{inf}}, "expanded_conv/depthwise/Relu6_bias#5");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*32*16);
    auto weights1 = om.constantInt(weightsData1,{1,1,32,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.0020443268585950136},{-0.27914440631866455},{0.2421589344739914}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#7");
    auto conv1 = om.conv(bias_cd0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#173");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{8.016967512958217e-06},{-inf},{inf}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#8");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*16*96);
    auto weights2 = om.constantInt(weightsData2,{1,1,16,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.00239623268134892},{-0.2872050106525421},{0.3238343298435211}}, "expanded_conv_1/expand/Relu6_weights#10");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_1/expand/Relu6#174");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.8793982235365547e-05},{-inf},{inf}}, "expanded_conv_1/expand/Relu6_bias#11");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData1 = mv::utils::generateSequence<int64_t> (3*3*96*1);
    auto d_weights1 = om.constantInt(d_weightsData1,{3,3,96,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{142},{0.0023037735372781754},{-0.32606151700019836},{0.2614007592201233}}, "expanded_conv_1/depthwise/Relu6_weights#13");
    auto depthConv1 = om.depthwiseConv(bias_c2, d_weights1, {2, 2}, {0, 1, 0, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_1/depthwise/Relu6#175");

    std::vector<int64_t> biasd_WeightsData1 = mv::utils::generateSequence<int64_t> (96);
    auto biasdWeights1 = om.constantInt(biasd_WeightsData1,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.034406502905767e-06},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6_bias#14");
    auto bias_cd1 = om.bias(depthConv1, biasdWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*96*32);
    auto weights3 = om.constantInt(weightsData3,{1,1,96,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.003013054607436061},{-0.36564400792121887},{0.40268489718437195}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#16");
    auto conv3 = om.conv(bias_cd1, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#176");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1815900506917387e-05},{-inf},{inf}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#17");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*32*144);
    auto weights4 = om.constantInt(weightsData4,{1,1,32,144}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.0029819777701050043},{-0.3435925841331482},{0.4168117642402649}}, "expanded_conv_2/depthwise/Relu6_weights#19");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_2/depthwise/Relu6#177");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (144);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.338806189072784e-05},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_bias#20");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData2 = mv::utils::generateSequence<int64_t> (3*3*144*1);
    auto d_weights2 = om.constantInt(d_weightsData2,{3,3,144,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0026192497462034225},{-0.31191185116767883},{0.35599684715270996}}, "expanded_conv_2/depthwise/Relu6_1_weights#22");
    auto depthConv2 = om.depthwiseConv(bias_c4, d_weights2, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_2/depthwise/Relu6_1#178");

    std::vector<int64_t> biasd_WeightsData2 = mv::utils::generateSequence<int64_t> (144);
    auto biasdWeights2 = om.constantInt(biasd_WeightsData2,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0271567589370534e-05},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1_bias#23");
    auto bias_cd2 = om.bias(depthConv2, biasdWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*144*32);
    auto weights5 = om.constantInt(weightsData5,{1,1,144,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{142},{0.0029734554700553417},{-0.42331647872924805},{0.3349146842956543}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#25");
    auto conv5 = om.conv(bias_cd2, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#179");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.166060974355787e-05},{-inf},{inf}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#26");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise0 = om.add({bias_c3,bias_c5}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_2/add/Add#180");

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*32*144);
    auto weights6 = om.constantInt(weightsData6,{1,1,32,144}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0027636997401714325},{-0.369789183139801},{0.33495423197746277}}, "expanded_conv_3/expand/Relu6_weights#29");
    auto conv6 = om.conv(eltwise0, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_3/expand/Relu6#181");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (144);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.1676076357834972e-05},{-inf},{inf}}, "expanded_conv_3/expand/Relu6_bias#30");
    auto bias_c6 = om.bias(conv6, biasWeights6, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData3 = mv::utils::generateSequence<int64_t> (3*3*144*1);
    auto d_weights3 = om.constantInt(d_weightsData3,{3,3,144,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{147},{0.0027841234114021063},{-0.4088488519191742},{0.3011026084423065}}, "expanded_conv_3/depthwise/Relu6_weights#32");
    auto depthConv3 = om.depthwiseConv(bias_c6, d_weights3, {2, 2}, {0, 1, 0, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_3/depthwise/Relu6#182");

    std::vector<int64_t> biasd_WeightsData3 = mv::utils::generateSequence<int64_t> (144);
    auto biasdWeights3 = om.constantInt(biasd_WeightsData3,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0918131010839716e-05},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6_bias#33");
    auto bias_cd3 = om.bias(depthConv3, biasdWeights3, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*144*32);
    auto weights7 = om.constantInt(weightsData7,{1,1,144,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.002796994522213936},{-0.36320796608924866},{0.35002562403678894}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#35");
    auto conv7 = om.conv(bias_cd3, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#183");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.096860614779871e-05},{-inf},{inf}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#36");
    auto bias_c7 = om.bias(conv7, biasWeights7, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights8 = om.constantInt(weightsData8,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{136},{0.0028090770356357098},{-0.3831981420516968},{0.3331165015697479}}, "expanded_conv_4/expand/Relu6_weights#38");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_4/expand/Relu6#184");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2031976186553948e-05},{-inf},{inf}}, "expanded_conv_4/expand/Relu6_bias#39");
    auto bias_c8 = om.bias(conv8, biasWeights8, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData4 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights4 = om.constantInt(d_weightsData4,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.002653743838891387},{-0.33790725469589233},{0.3387974500656128}}, "expanded_conv_4/depthwise/Relu6_weights#41");
    auto depthConv4 = om.depthwiseConv(bias_c8, d_weights4, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_4/depthwise/Relu6#185");

    std::vector<int64_t> biasd_WeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights4 = om.constantInt(biasd_WeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0406838555354625e-05},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6_bias#42");
    auto bias_cd4 = om.bias(depthConv4, biasdWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights9 = om.constantInt(weightsData9,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0030288114212453365},{-0.3936502933502197},{0.37869659066200256}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#44");
    auto conv9 = om.conv(bias_cd4, weights9, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#186");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.187769157695584e-05},{-inf},{inf}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#45");
    auto bias_c9 = om.bias(conv9, biasWeights9, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise1 = om.add({bias_c7,bias_c9}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_4/add/Add#187");

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights10 = om.constantInt(weightsData10,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0028022583574056625},{-0.3673075735569},{0.347268283367157}}, "expanded_conv_5/expand/Relu6_weights#48");
    auto conv10 = om.conv(eltwise1, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_5/expand/Relu6#188");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.1978496079100296e-05},{-inf},{inf}}, "expanded_conv_5/expand/Relu6_bias#49");
    auto bias_c10 = om.bias(conv10, biasWeights10, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData5 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights5 = om.constantInt(d_weightsData5,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.002570582553744316},{-0.3373497426509857},{0.31814879179000854}}, "expanded_conv_5/depthwise/Relu6_weights#51");
    auto depthConv5 = om.depthwiseConv(bias_c10, d_weights5, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_5/depthwise/Relu6#189");

    std::vector<int64_t> biasd_WeightsData5 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights5 = om.constantInt(biasd_WeightsData5,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0080715583171695e-05},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6_bias#52");
    auto bias_cd5 = om.bias(depthConv5, biasdWeights5, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights11 = om.constantInt(weightsData11,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.003027612343430519},{-0.3664873242378235},{0.40555381774902344}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#54");
    auto conv11 = om.conv(bias_cd5, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#190");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1872989489347674e-05},{-inf},{inf}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#55");
    auto bias_c11 = om.bias(conv11, biasWeights11, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise2 = om.add({eltwise1,bias_c11}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_5/add/Add#191");

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights12 = om.constantInt(weightsData12,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.002645136322826147},{-0.34687551856040955},{0.3276342451572418}}, "expanded_conv_6/expand/Relu6_weights#58");
    auto conv12 = om.conv(eltwise2, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_6/expand/Relu6#192");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.074616713798605e-05},{-inf},{inf}}, "expanded_conv_6/expand/Relu6_bias#59");
    auto bias_c12 = om.bias(conv12, biasWeights12, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData6 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights6 = om.constantInt(d_weightsData6,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.002544405870139599},{-0.30167388916015625},{0.3471496105194092}}, "expanded_conv_6/depthwise/Relu6_weights#61");
    auto depthConv6 = om.depthwiseConv(bias_c12, d_weights6, {2, 2}, {0, 1, 0, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_6/depthwise/Relu6#193");

    std::vector<int64_t> biasd_WeightsData6 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights6 = om.constantInt(biasd_WeightsData6,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.978061825677287e-06},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6_bias#62");
    auto bias_cd6 = om.bias(depthConv6, biasdWeights6, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (1*1*192*64);
    auto weights13 = om.constantInt(weightsData13,{1,1,192,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0030932214576750994},{-0.4181155264377594},{0.37065595388412476}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#64");
    auto conv13 = om.conv(bias_cd6, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#194");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2130280083511025e-05},{-inf},{inf}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#65");
    auto bias_c13 = om.bias(conv13, biasWeights13, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights14 = om.constantInt(weightsData14,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.003283844096586108},{-0.4013615548610687},{0.436018705368042}}, "expanded_conv_7/expand/Relu6_weights#67");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_7/expand/Relu6#195");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.5755640308489092e-05},{-inf},{inf}}, "expanded_conv_7/expand/Relu6_bias#68");
    auto bias_c14 = om.bias(conv14, biasWeights14, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData7 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights7 = om.constantInt(d_weightsData7,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0026749272365123034},{-0.3373517692089081},{0.3447546660900116}}, "expanded_conv_7/depthwise/Relu6_weights#70");
    auto depthConv7 = om.depthwiseConv(bias_c14, d_weights7, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_7/depthwise/Relu6#196");

    std::vector<int64_t> biasd_WeightsData7 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights7 = om.constantInt(biasd_WeightsData7,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0489910891919862e-05},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6_bias#71");
    auto bias_cd7 = om.bias(depthConv7, biasdWeights7, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights15 = om.constantInt(weightsData15,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.00342287658713758},{-0.41974323987960815},{0.45309028029441833}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#73");
    auto conv15 = om.conv(bias_cd7, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#197");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3423044947558083e-05},{-inf},{inf}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#74");
    auto bias_c15 = om.bias(conv15, biasWeights15, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise3 = om.add({bias_c13,bias_c15}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_7/add/Add#198");

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights16 = om.constantInt(weightsData16,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003092024242505431},{-0.38333263993263245},{0.40513351559638977}}, "expanded_conv_8/expand/Relu6_weights#77");
    auto conv16 = om.conv(eltwise3, weights16, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_8/expand/Relu6#199");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4251170543720946e-05},{-inf},{inf}}, "expanded_conv_8/expand/Relu6_bias#78");
    auto bias_c16 = om.bias(conv16, biasWeights16, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData8 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights8 = om.constantInt(d_weightsData8,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{138},{0.0029074354097247124},{-0.4021194577217102},{0.3392765522003174}}, "expanded_conv_8/depthwise/Relu6_weights#80");
    auto depthConv8 = om.depthwiseConv(bias_c16, d_weights8, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_8/depthwise/Relu6#200");

    std::vector<int64_t> biasd_WeightsData8 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights8 = om.constantInt(biasd_WeightsData8,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1401707524782978e-05},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6_bias#81");
    auto bias_cd8 = om.bias(depthConv8, biasdWeights8, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights17 = om.constantInt(weightsData17,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0031425657216459513},{-0.3783290386199951},{0.4230251908302307}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#83");
    auto conv17 = om.conv(bias_cd8, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#201");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2323786904744338e-05},{-inf},{inf}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#84");
    auto bias_c17 = om.bias(conv17, biasWeights17, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise4 = om.add({eltwise3,bias_c17}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_8/add/Add#202");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights18 = om.constantInt(weightsData18,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003101628040894866},{-0.4189969599246979},{0.3719181716442108}}, "expanded_conv_9/expand/Relu6_weights#87");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_9/expand/Relu6#203");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4326493075932376e-05},{-inf},{inf}}, "expanded_conv_9/expand/Relu6_bias#88");
    auto bias_c18 = om.bias(conv18, biasWeights18, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData9 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights9 = om.constantInt(d_weightsData9,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0027253320440649986},{-0.3462289273738861},{0.34873074293136597}}, "expanded_conv_9/depthwise/Relu6_weights#90");
    auto depthConv9 = om.depthwiseConv(bias_c18, d_weights9, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_9/depthwise/Relu6#204");

    std::vector<int64_t> biasd_WeightsData9 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights9 = om.constantInt(biasd_WeightsData9,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0687576832424384e-05},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6_bias#91");
    auto bias_cd9 = om.bias(depthConv9, biasdWeights9, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights19 = om.constantInt(weightsData19,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003123433794826269},{-0.38817867636680603},{0.40829694271087646}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#93");
    auto conv19 = om.conv(bias_cd9, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#205");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2248759958310984e-05},{-inf},{inf}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#94");
    auto bias_c19 = om.bias(conv19, biasWeights19, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise5 = om.add({eltwise4,bias_c19}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_9/add/Add#206");

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights20 = om.constantInt(weightsData20,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0030837294179946184},{-0.3856784403324127},{0.40067258477211}}, "expanded_conv_10/expand/Relu6_weights#97");
    auto conv20 = om.conv(eltwise5, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_10/expand/Relu6#207");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4186114387703128e-05},{-inf},{inf}}, "expanded_conv_10/expand/Relu6_bias#98");
    auto bias_c20 = om.bias(conv20, biasWeights20, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData10 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights10 = om.constantInt(d_weightsData10,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0027900137938559055},{-0.33103904128074646},{0.3804144561290741}}, "expanded_conv_10/depthwise/Relu6_weights#100");
    auto depthConv10 = om.depthwiseConv(bias_c20, d_weights10, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_10/depthwise/Relu6#208");

    std::vector<int64_t> biasd_WeightsData10 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights10 = om.constantInt(biasd_WeightsData10,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0941230357275344e-05},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6_bias#101");
    auto bias_cd10 = om.bias(depthConv10, biasdWeights10, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*384*96);
    auto weights21 = om.constantInt(weightsData21,{1,1,384,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0034912910778075457},{-0.42566612362861633},{0.4646131098270416}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#103");
    auto conv21 = om.conv(bias_cd10, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#209");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.369133769912878e-05},{-inf},{inf}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#104");
    auto bias_c21 = om.bias(conv21, biasWeights21, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights22 = om.constantInt(weightsData22,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.003542622085660696},{-0.4422760009765625},{0.46109265089035034}}, "expanded_conv_11/expand/Relu6_weights#106");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_11/expand/Relu6#210");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.7785272322944365e-05},{-inf},{inf}}, "expanded_conv_11/expand/Relu6_bias#107");
    auto bias_c22 = om.bias(conv22, biasWeights22, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData11 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights11 = om.constantInt(d_weightsData11,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0030674629379063845},{-0.41154029965400696},{0.37066277861595154}}, "expanded_conv_11/depthwise/Relu6_weights#109");
    auto depthConv11 = om.depthwiseConv(bias_c22, d_weights11, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_11/depthwise/Relu6#211");

    std::vector<int64_t> biasd_WeightsData11 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights11 = om.constantInt(biasd_WeightsData11,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2029267054458614e-05},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6_bias#110");
    auto bias_cd11 = om.bias(depthConv11, biasdWeights11, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*576*96);
    auto weights23 = om.constantInt(weightsData23,{1,1,576,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0035177813842892647},{-0.43919509649276733},{0.45783916115760803}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#112");
    auto conv23 = om.conv(bias_cd11, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#212");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3795221093459986e-05},{-inf},{inf}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#113");
    auto bias_c23 = om.bias(conv23, biasWeights23, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise6 = om.add({bias_c21,bias_c23}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_11/add/Add#213");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights24 = om.constantInt(weightsData24,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0033828995656222105},{-0.4039537012577057},{0.4586856961250305}}, "expanded_conv_12/expand/Relu6_weights#116");
    auto conv24 = om.conv(eltwise6, weights24, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_12/expand/Relu6#214");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.6532545234658755e-05},{-inf},{inf}}, "expanded_conv_12/expand/Relu6_bias#117");
    auto bias_c24 = om.bias(conv24, biasWeights24, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData12 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights12 = om.constantInt(d_weightsData12,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.002697212155908346},{-0.332965612411499},{0.3548234701156616}}, "expanded_conv_12/depthwise/Relu6_weights#119");
    auto depthConv12 = om.depthwiseConv(bias_c24, d_weights12, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_12/depthwise/Relu6#215");

    std::vector<int64_t> biasd_WeightsData12 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights12 = om.constantInt(biasd_WeightsData12,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.057730241882382e-05},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6_bias#120");
    auto bias_cd12 = om.bias(depthConv12, biasdWeights12, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*576*96);
    auto weights25 = om.constantInt(weightsData25,{1,1,576,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0033039639238268137},{-0.4030507504940033},{0.43946003913879395}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#122");
    auto conv25 = om.conv(bias_cd12, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#216");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2956721548107453e-05},{-inf},{inf}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#123");
    auto bias_c25 = om.bias(conv25, biasWeights25, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise7 = om.add({eltwise6,bias_c25}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_12/add/Add#217");

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights26 = om.constantInt(weightsData26,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003184879431501031},{-0.4049041271209717},{0.4072401225566864}}, "expanded_conv_13/expand/Relu6_weights#126");
    auto conv26 = om.conv(eltwise7, weights26, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_13/expand/Relu6#218");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4979446607176214e-05},{-inf},{inf}}, "expanded_conv_13/expand/Relu6_bias#127");
    auto bias_c26 = om.bias(conv26, biasWeights26, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData13 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights13 = om.constantInt(d_weightsData13,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.00257533835247159},{-0.3389812111854553},{0.31773003935813904}}, "expanded_conv_13/depthwise/Relu6_weights#129");
    auto depthConv13 = om.depthwiseConv(bias_c26, d_weights13, {2, 2}, {0, 1, 0, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_13/depthwise/Relu6#219");

    std::vector<int64_t> biasd_WeightsData13 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights13 = om.constantInt(biasd_WeightsData13,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0099365681526251e-05},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6_bias#130");
    auto bias_cd13 = om.bias(depthConv13, biasdWeights13, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*576*160);
    auto weights27 = om.constantInt(weightsData27,{1,1,576,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.003236441407352686},{-0.4176654815673828},{0.4076271057128906}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#132");
    auto conv27 = om.conv(bias_cd13, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#220");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2691927622654475e-05},{-inf},{inf}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#133");
    auto bias_c27 = om.bias(conv27, biasWeights27, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights28 = om.constantInt(weightsData28,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0033436305820941925},{-0.4370753765106201},{0.4155504107475281}}, "expanded_conv_14/expand/Relu6_weights#135");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_14/expand/Relu6#221");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.622455394885037e-05},{-inf},{inf}}, "expanded_conv_14/expand/Relu6_bias#136");
    auto bias_c28 = om.bias(conv28, biasWeights28, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData14 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights14 = om.constantInt(d_weightsData14,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0030434317886829376},{-0.39223265647888184},{0.38384243845939636}}, "expanded_conv_14/depthwise/Relu6_weights#138");
    auto depthConv14 = om.depthwiseConv(bias_c28, d_weights14, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_14/depthwise/Relu6#222");

    std::vector<int64_t> biasd_WeightsData14 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights14 = om.constantInt(biasd_WeightsData14,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1935026122955605e-05},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6_bias#139");
    auto bias_cd14 = om.bias(depthConv14, biasdWeights14, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (1*1*960*160);
    auto weights29 = om.constantInt(weightsData29,{1,1,960,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.003733013290911913},{-0.4963448643684387},{0.45557352900505066}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#141");
    auto conv29 = om.conv(bias_cd14, weights29, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#223");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4639267646998633e-05},{-inf},{inf}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#142");
    auto bias_c29 = om.bias(conv29, biasWeights29, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise8 = om.add({bias_c27,bias_c29}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_14/add/Add#224");

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights30 = om.constantInt(weightsData30,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.00334560452029109},{-0.4360058009624481},{0.41712334752082825}}, "expanded_conv_15/expand/Relu6_weights#145");
    auto conv30 = om.conv(eltwise8, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_15/expand/Relu6#225");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.624003536766395e-05},{-inf},{inf}}, "expanded_conv_15/expand/Relu6_bias#146");
    auto bias_c30 = om.bias(conv30, biasWeights30, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData15 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights15 = om.constantInt(d_weightsData15,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0029716617427766323},{-0.38779741525650024},{0.36997634172439575}}, "expanded_conv_15/depthwise/Relu6_weights#148");
    auto depthConv15 = om.depthwiseConv(bias_c30, d_weights15, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_15/depthwise/Relu6#226");

    std::vector<int64_t> biasd_WeightsData15 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights15 = om.constantInt(biasd_WeightsData15,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1653575711534359e-05},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6_bias#149");
    auto bias_cd15 = om.bias(depthConv15, biasdWeights15, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*960*160);
    auto weights31 = om.constantInt(weightsData31,{1,1,960,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0034228439908474684},{-0.43593573570251465},{0.43688949942588806}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#151");
    auto conv31 = om.conv(bias_cd15, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#227");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3422917618299834e-05},{-inf},{inf}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#152");
    auto bias_c31 = om.bias(conv31, biasWeights31, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise9 = om.add({eltwise8,bias_c31}, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_15/add/Add#228");

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights32 = om.constantInt(weightsData32,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0036992442328482866},{-0.49393245577812195},{0.449374794960022}}, "expanded_conv_16/expand/Relu6_weights#155");
    auto conv32 = om.conv(eltwise9, weights32, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_16/expand/Relu6#229");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.9013679522904567e-05},{-inf},{inf}}, "expanded_conv_16/expand/Relu6_bias#156");
    auto bias_c32 = om.bias(conv32, biasWeights32, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData16 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights16 = om.constantInt(d_weightsData16,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.003115250263363123},{-0.41396597027778625},{0.3804228603839874}}, "expanded_conv_16/depthwise/Relu6_weights#158");
    auto depthConv16 = om.depthwiseConv(bias_c32, d_weights16, {1, 1}, {1, 1, 1, 1}, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_16/depthwise/Relu6#230");

    std::vector<int64_t> biasd_WeightsData16 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights16 = om.constantInt(biasd_WeightsData16,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2216667528264225e-05},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6_bias#159");
    auto bias_cd16 = om.bias(depthConv16, biasdWeights16, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*960*320);
    auto weights33 = om.constantInt(weightsData33,{1,1,960,320}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0035930087324231863},{-0.4399093687534332},{0.47630783915519714}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd_weights#161");
    auto conv33 = om.conv(bias_cd16, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#231");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (320);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{320}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4090230251895264e-05},{-inf},{inf}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd_bias#162");
    auto bias_c33 = om.bias(conv33, biasWeights33, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*320*1280);
    auto weights34 = om.constantInt(weightsData34,{1,1,320,1280}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003499699058011174},{-0.44588136672973633},{0.4465419054031372}}, "Conv_1/Relu6_weights#164");
    auto conv34 = om.conv(bias_c33, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "Conv_1/Relu6#232");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (1280);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{1280}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.7448621040093713e-05},{-inf},{inf}}, "Conv_1/Relu6_bias#165");
    auto bias_c34 = om.bias(conv34, biasWeights34, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.averagePool(bias_c34, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{0},{0.003921568859368563},{0.0},{1.0}}, "Logits/AvgPool/AvgPool#233");

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (1*1*1280*1024);
    auto weights35 = om.constantInt(weightsData35,{1,1,1280,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.003863171674311161},{-0.5084267854690552},{0.47668200731277466}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd_weights#168");
    auto conv35 = om.conv(pool0, weights35, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd#234");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5149693354032934e-05},{-inf},{inf}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd_bias#169");
    auto bias_c35 = om.bias(conv35, biasWeights35, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c35);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/mobilenet_streaming.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}