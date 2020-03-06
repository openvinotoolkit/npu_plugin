#include <iostream>
#include <fstream>
#include <unistd.h>

//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"


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
    fprintf(stderr, "./mobile_net -d {comp_descriptor}\n");
  }

  const char *comp_descriptor_;
};  // struct InputParams //



int main(int argc, char **argv)
{

    InputParams params;

    if (!params.parse_args(argc, argv)) { return -1; }

    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*32);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "Conv/Relu6#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.076643613690976e-05},{-inf},{inf}}, "Conv/Relu6#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (3*3*32*1);
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,32,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.002440826501697302},{-0.27914440631866455},{0.3432663381099701}}, "expanded_conv/depthwise/Relu6#3_weights#4");
    auto depthConv0 = om.depthwiseConv(bias_c0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv/depthwise/Relu6#172");

    std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.571868758939672e-06},{-inf},{inf}}, "expanded_conv/depthwise/Relu6#3_bias#5");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*32*16);
    auto weights1 = om.constantInt(weightsData1,{1,1,32,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.002668388420715928},{-0.3064141273498535},{0.374024897813797}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#6_weights#7");
    auto conv1 = om.conv(bias_cd0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#173");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0464267688803375e-05},{-inf},{inf}}, "expanded_conv/project/BatchNorm/FusedBatchNorm/BiasAdd#6_bias#8");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*16*96);
    auto weights2 = om.constantInt(weightsData2,{1,1,16,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0024104418698698282},{-0.32530343532562256},{0.2893592417240143}}, "expanded_conv_1/expand/Relu6#9_weights#10");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_1/expand/Relu6#174");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.890542625915259e-05},{-inf},{inf}}, "expanded_conv_1/expand/Relu6#9_bias#11");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData1 = mv::utils::generateSequence<int64_t> (3*3*96*1);
    auto d_weights1 = om.constantInt(d_weightsData1,{3,3,96,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{118},{0.002482198877260089},{-0.29253050684928894},{0.34043022990226746}}, "expanded_conv_1/depthwise/Relu6#12_weights#13");
    auto depthConv1 = om.depthwiseConv(bias_c2, d_weights1, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_1/depthwise/Relu6#175");

    std::vector<int64_t> biasd_WeightsData1 = mv::utils::generateSequence<int64_t> (96);
    auto biasdWeights1 = om.constantInt(biasd_WeightsData1,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.734113518788945e-06},{-inf},{inf}}, "expanded_conv_1/depthwise/Relu6#12_bias#14");
    auto bias_cd1 = om.bias(depthConv1, biasdWeights1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*96*32);
    auto weights3 = om.constantInt(weightsData3,{1,1,96,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0028508694376796484},{-0.36564400792121887},{0.3613277077674866}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#15_weights#16");
    auto conv3 = om.conv(bias_cd1, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#176");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1179879948031157e-05},{-inf},{inf}}, "expanded_conv_1/project/BatchNorm/FusedBatchNorm/BiasAdd#15_bias#17");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights4 = om.constantInt(weightsData4,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.0029819777701050043},{-0.3435925841331482},{0.4168117642402649}}, "expanded_conv_2/depthwise/Relu6#18_weights#19");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_2/depthwise/Relu6#177");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.338806189072784e-05},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6#18_bias#20");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData2 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights2 = om.constantInt(d_weightsData2,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{142},{0.0029706729110330343},{-0.42331647872924805},{0.33420512080192566}}, "expanded_conv_2/depthwise/Relu6_1#21_weights#22");
    auto depthConv2 = om.depthwiseConv(bias_c4, d_weights2, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_2/depthwise/Relu6_1#178");

    std::vector<int64_t> biasd_WeightsData2 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights2 = om.constantInt(biasd_WeightsData2,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1649697626125999e-05},{-inf},{inf}}, "expanded_conv_2/depthwise/Relu6_1#21_bias#23");
    auto bias_cd2 = om.bias(depthConv2, biasdWeights2, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights5 = om.constantInt(weightsData5,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.00287077110260725},{-0.369789183139801},{0.3622574508190155}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#24_weights#25");
    auto conv5 = om.conv(bias_cd2, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#179");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1257925507379696e-05},{-inf},{inf}}, "expanded_conv_2/project/BatchNorm/FusedBatchNorm/BiasAdd#24_bias#26");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise0 = om.add({bias_c3,bias_c5}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_2/add/Add#180");

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights6 = om.constantInt(weightsData6,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.002975978422909975},{-0.4088488519191742},{0.35002562403678894}}, "expanded_conv_3/expand/Relu6#28_weights#29");
    auto conv6 = om.conv(eltwise0, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_3/expand/Relu6#181");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.3341006453847513e-05},{-inf},{inf}}, "expanded_conv_3/expand/Relu6#28_bias#30");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData3 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights3 = om.constantInt(d_weightsData3,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.002540246583521366},{-0.288228839635849},{0.35953405499458313}}, "expanded_conv_3/depthwise/Relu6#31_weights#32");
    auto depthConv3 = om.depthwiseConv(bias_c6, d_weights3, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_3/depthwise/Relu6#182");

    std::vector<int64_t> biasd_WeightsData3 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights3 = om.constantInt(biasd_WeightsData3,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.961751857190393e-06},{-inf},{inf}}, "expanded_conv_3/depthwise/Relu6#31_bias#33");
    auto bias_cd3 = om.bias(depthConv3, biasdWeights3, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights7 = om.constantInt(weightsData7,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.002676468575373292},{-0.34124860167503357},{0.3412508964538574}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#34_weights#35");
    auto conv7 = om.conv(bias_cd3, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#183");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0495955393707845e-05},{-inf},{inf}}, "expanded_conv_3/project/BatchNorm/FusedBatchNorm/BiasAdd#34_bias#36");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights8 = om.constantInt(weightsData8,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0030288114212453365},{-0.3936502933502197},{0.37869659066200256}}, "expanded_conv_4/expand/Relu6#37_weights#38");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_4/expand/Relu6#184");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.375538315391168e-05},{-inf},{inf}}, "expanded_conv_4/expand/Relu6#37_bias#39");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData4 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights4 = om.constantInt(d_weightsData4,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0028959375340491533},{-0.3905291259288788},{0.34793493151664734}}, "expanded_conv_4/depthwise/Relu6#40_weights#41");
    auto depthConv4 = om.depthwiseConv(bias_c8, d_weights4, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_4/depthwise/Relu6#185");

    std::vector<int64_t> biasd_WeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights4 = om.constantInt(biasd_WeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1356617505953182e-05},{-inf},{inf}}, "expanded_conv_4/depthwise/Relu6#40_bias#42");
    auto bias_cd4 = om.bias(depthConv4, biasdWeights4, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights9 = om.constantInt(weightsData9,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.002877321559935808},{-0.3673075735569},{0.36640942096710205}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#43_weights#44");
    auto conv9 = om.conv(bias_cd4, weights9, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#186");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1283614185231272e-05},{-inf},{inf}}, "expanded_conv_4/project/BatchNorm/FusedBatchNorm/BiasAdd#43_bias#45");
    auto bias_c9 = om.bias(conv9, biasWeights9, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise1 = om.add({bias_c7,bias_c9}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_4/add/Add#187");

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights10 = om.constantInt(weightsData10,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.003027612343430519},{-0.3664873242378235},{0.40555381774902344}}, "expanded_conv_5/expand/Relu6#47_weights#48");
    auto conv10 = om.conv(eltwise1, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_5/expand/Relu6#188");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.3745978978695348e-05},{-inf},{inf}}, "expanded_conv_5/expand/Relu6#47_bias#49");
    auto bias_c10 = om.bias(conv10, biasWeights10, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData5 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights5 = om.constantInt(d_weightsData5,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.0027801841497421265},{-0.3697494864463806},{0.3391974866390228}}, "expanded_conv_5/depthwise/Relu6#50_weights#51");
    auto depthConv5 = om.depthwiseConv(bias_c10, d_weights5, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_5/depthwise/Relu6#189");

    std::vector<int64_t> biasd_WeightsData5 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights5 = om.constantInt(biasd_WeightsData5,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0902683243330102e-05},{-inf},{inf}}, "expanded_conv_5/depthwise/Relu6#50_bias#52");
    auto bias_cd5 = om.bias(depthConv5, biasdWeights5, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights11 = om.constantInt(weightsData11,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.002501942217350006},{-0.3187469244003296},{0.31924834847450256}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#53_weights#54");
    auto conv11 = om.conv(bias_cd5, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#190");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.811537893256173e-06},{-inf},{inf}}, "expanded_conv_5/project/BatchNorm/FusedBatchNorm/BiasAdd#53_bias#55");
    auto bias_c11 = om.bias(conv11, biasWeights11, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise2 = om.add({eltwise1,bias_c11}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_5/add/Add#191");

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights12 = om.constantInt(weightsData12,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.003060962539166212},{-0.38541027903556824},{0.39513516426086426}}, "expanded_conv_6/expand/Relu6#57_weights#58");
    auto conv12 = om.conv(eltwise2, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_6/expand/Relu6#192");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4007549654925242e-05},{-inf},{inf}}, "expanded_conv_6/expand/Relu6#57_bias#59");
    auto bias_c12 = om.bias(conv12, biasWeights12, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData6 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights6 = om.constantInt(d_weightsData6,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{146},{0.0028609014116227627},{-0.4181155264377594},{0.31141436100006104}}, "expanded_conv_6/depthwise/Relu6#60_weights#61");
    auto depthConv6 = om.depthwiseConv(bias_c12, d_weights6, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_6/depthwise/Relu6#193");

    std::vector<int64_t> biasd_WeightsData6 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights6 = om.constantInt(biasd_WeightsData6,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1219221960345749e-05},{-inf},{inf}}, "expanded_conv_6/depthwise/Relu6#60_bias#62");
    auto bias_cd6 = om.bias(depthConv6, biasdWeights6, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (1*1*192*64);
    auto weights13 = om.constantInt(weightsData13,{1,1,192,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.003011964727193117},{-0.3685387074947357},{0.39951232075691223}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#63_weights#64");
    auto conv13 = om.conv(bias_cd6, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#194");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1811626791313756e-05},{-inf},{inf}}, "expanded_conv_6/project/BatchNorm/FusedBatchNorm/BiasAdd#63_bias#65");
    auto bias_c13 = om.bias(conv13, biasWeights13, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights14 = om.constantInt(weightsData14,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.0032514804042875767},{-0.3931088149547577},{0.436018705368042}}, "expanded_conv_7/expand/Relu6#66_weights#67");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_7/expand/Relu6#195");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.550180761318188e-05},{-inf},{inf}}, "expanded_conv_7/expand/Relu6#66_bias#68");
    auto bias_c14 = om.bias(conv14, biasWeights14, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData7 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights7 = om.constantInt(d_weightsData7,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{144},{0.00290895183570683},{-0.41974323987960815},{0.3220394551753998}}, "expanded_conv_7/depthwise/Relu6#69_weights#70");
    auto depthConv7 = om.depthwiseConv(bias_c14, d_weights7, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_7/depthwise/Relu6#196");

    std::vector<int64_t> biasd_WeightsData7 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights7 = om.constantInt(biasd_WeightsData7,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.140765380114317e-05},{-inf},{inf}}, "expanded_conv_7/depthwise/Relu6#69_bias#71");
    auto bias_cd7 = om.bias(depthConv7, biasdWeights7, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights15 = om.constantInt(weightsData15,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0033494089730083942},{-0.4010090231895447},{0.45309028029441833}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#72_weights#73");
    auto conv15 = om.conv(bias_cd7, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#197");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3134937034919858e-05},{-inf},{inf}}, "expanded_conv_7/project/BatchNorm/FusedBatchNorm/BiasAdd#72_bias#74");
    auto bias_c15 = om.bias(conv15, biasWeights15, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise3 = om.add({bias_c13,bias_c15}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_7/add/Add#198");

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights16 = om.constantInt(weightsData16,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.002865646732971072},{-0.36032533645629883},{0.3704145848751068}}, "expanded_conv_8/expand/Relu6#76_weights#77");
    auto conv16 = om.conv(eltwise3, weights16, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_8/expand/Relu6#199");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2475660443888046e-05},{-inf},{inf}}, "expanded_conv_8/expand/Relu6#76_bias#78");
    auto bias_c16 = om.bias(conv16, biasWeights16, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData8 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights8 = om.constantInt(d_weightsData8,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0031425657216459513},{-0.3783290386199951},{0.4230251908302307}}, "expanded_conv_8/depthwise/Relu6#79_weights#80");
    auto depthConv8 = om.depthwiseConv(bias_c16, d_weights8, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_8/depthwise/Relu6#200");

    std::vector<int64_t> biasd_WeightsData8 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights8 = om.constantInt(biasd_WeightsData8,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2323786904744338e-05},{-inf},{inf}}, "expanded_conv_8/depthwise/Relu6#79_bias#81");
    auto bias_cd8 = om.bias(depthConv8, biasdWeights8, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights17 = om.constantInt(weightsData17,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0028650660533457994},{-0.35315361618995667},{0.3774382174015045}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#82_weights#83");
    auto conv17 = om.conv(bias_cd8, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#201");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1235552847210784e-05},{-inf},{inf}}, "expanded_conv_8/project/BatchNorm/FusedBatchNorm/BiasAdd#82_bias#84");
    auto bias_c17 = om.bias(conv17, biasWeights17, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise4 = om.add({eltwise3,bias_c17}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_8/add/Add#202");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights18 = om.constantInt(weightsData18,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003101628040894866},{-0.4189969599246979},{0.3719181716442108}}, "expanded_conv_9/expand/Relu6#86_weights#87");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_9/expand/Relu6#203");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4326493075932376e-05},{-inf},{inf}}, "expanded_conv_9/expand/Relu6#86_bias#88");
    auto bias_c18 = om.bias(conv18, biasWeights18, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData9 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights9 = om.constantInt(d_weightsData9,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0025852471590042114},{-0.33233240246772766},{0.32690560817718506}}, "expanded_conv_9/depthwise/Relu6#89_weights#90");
    auto depthConv9 = om.depthwiseConv(bias_c18, d_weights9, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_9/depthwise/Relu6#204");

    std::vector<int64_t> biasd_WeightsData9 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights9 = om.constantInt(biasd_WeightsData9,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.01382238426595e-05},{-inf},{inf}}, "expanded_conv_9/depthwise/Relu6#89_bias#91");
    auto bias_cd9 = om.bias(depthConv9, biasdWeights9, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (1*1*384*64);
    auto weights19 = om.constantInt(weightsData19,{1,1,384,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0031859634909778833},{-0.40299081802368164},{0.4094298481941223}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#92_weights#93");
    auto conv19 = om.conv(bias_cd9, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#205");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2493974281824194e-05},{-inf},{inf}}, "expanded_conv_9/project/BatchNorm/FusedBatchNorm/BiasAdd#92_bias#94");
    auto bias_c19 = om.bias(conv19, biasWeights19, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise5 = om.add({eltwise4,bias_c19}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_9/add/Add#206");

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*64*384);
    auto weights20 = om.constantInt(weightsData20,{1,1,64,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0031280340626835823},{-0.3753364384174347},{0.4223122298717499}}, "expanded_conv_10/expand/Relu6#96_weights#97");
    auto conv20 = om.conv(eltwise5, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_10/expand/Relu6#207");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.45335995714413e-05},{-inf},{inf}}, "expanded_conv_10/expand/Relu6#96_bias#98");
    auto bias_c20 = om.bias(conv20, biasWeights20, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData10 = mv::utils::generateSequence<int64_t> (3*3*384*1);
    auto d_weights10 = om.constantInt(d_weightsData10,{3,3,384,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.002815549960359931},{-0.3354185223579407},{0.38254669308662415}}, "expanded_conv_10/depthwise/Relu6#99_weights#100");
    auto depthConv10 = om.depthwiseConv(bias_c20, d_weights10, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_10/depthwise/Relu6#208");

    std::vector<int64_t> biasd_WeightsData10 = mv::utils::generateSequence<int64_t> (384);
    auto biasdWeights10 = om.constantInt(biasd_WeightsData10,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1041372090403456e-05},{-inf},{inf}}, "expanded_conv_10/depthwise/Relu6#99_bias#101");
    auto bias_cd10 = om.bias(depthConv10, biasdWeights10, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*384*96);
    auto weights21 = om.constantInt(weightsData21,{1,1,384,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.003169028088450432},{-0.4045146107673645},{0.40358757972717285}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#102_weights#103");
    auto conv21 = om.conv(bias_cd10, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#209");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2427561159711331e-05},{-inf},{inf}}, "expanded_conv_10/project/BatchNorm/FusedBatchNorm/BiasAdd#102_bias#104");
    auto bias_c21 = om.bias(conv21, biasWeights21, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights22 = om.constantInt(weightsData22,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0034567732363939285},{-0.4203844964504242},{0.46109265089035034}}, "expanded_conv_11/expand/Relu6#105_weights#106");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_11/expand/Relu6#210");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.7111946110380813e-05},{-inf},{inf}}, "expanded_conv_11/expand/Relu6#105_bias#107");
    auto bias_c22 = om.bias(conv22, biasWeights22, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData11 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights11 = om.constantInt(d_weightsData11,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.00275207101367414},{-0.3507096767425537},{0.3510684370994568}}, "expanded_conv_11/depthwise/Relu6#108_weights#109");
    auto depthConv11 = om.depthwiseConv(bias_c22, d_weights11, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_11/depthwise/Relu6#211");

    std::vector<int64_t> biasd_WeightsData11 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights11 = om.constantInt(biasd_WeightsData11,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.079243520507589e-05},{-inf},{inf}}, "expanded_conv_11/depthwise/Relu6#108_bias#110");
    auto bias_cd11 = om.bias(depthConv11, biasdWeights11, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*576*96);
    auto weights23 = om.constantInt(weightsData23,{1,1,576,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0035177813842892647},{-0.43919509649276733},{0.45783916115760803}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#111_weights#112");
    auto conv23 = om.conv(bias_cd11, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#212");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3795221093459986e-05},{-inf},{inf}}, "expanded_conv_11/project/BatchNorm/FusedBatchNorm/BiasAdd#111_bias#113");
    auto bias_c23 = om.bias(conv23, biasWeights23, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise6 = om.add({bias_c21,bias_c23}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_11/add/Add#213");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights24 = om.constantInt(weightsData24,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003230717731639743},{-0.41139814257621765},{0.4124348759651184}}, "expanded_conv_12/expand/Relu6#115_weights#116");
    auto conv24 = om.conv(eltwise6, weights24, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_12/expand/Relu6#214");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.533896258682944e-05},{-inf},{inf}}, "expanded_conv_12/expand/Relu6#115_bias#117");
    auto bias_c24 = om.bias(conv24, biasWeights24, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData12 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights12 = om.constantInt(d_weightsData12,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0029778408352285624},{-0.4030507504940033},{0.35629868507385254}}, "expanded_conv_12/depthwise/Relu6#118_weights#119");
    auto depthConv12 = om.depthwiseConv(bias_c24, d_weights12, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_12/depthwise/Relu6#215");

    std::vector<int64_t> biasd_WeightsData12 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights12 = om.constantInt(biasd_WeightsData12,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1677807378873695e-05},{-inf},{inf}}, "expanded_conv_12/depthwise/Relu6#118_bias#120");
    auto bias_cd12 = om.bias(depthConv12, biasdWeights12, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*576*96);
    auto weights25 = om.constantInt(weightsData25,{1,1,576,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.0032903472892940044},{-0.3995785415172577},{0.43946003913879395}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#121_weights#122");
    auto conv25 = om.conv(bias_cd12, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#216");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.290332329517696e-05},{-inf},{inf}}, "expanded_conv_12/project/BatchNorm/FusedBatchNorm/BiasAdd#121_bias#123");
    auto bias_c25 = om.bias(conv25, biasWeights25, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise7 = om.add({eltwise6,bias_c25}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_12/add/Add#217");

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (1*1*96*576);
    auto weights26 = om.constantInt(weightsData26,{1,1,96,576}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003184879431501031},{-0.4049041271209717},{0.4072401225566864}}, "expanded_conv_13/expand/Relu6#125_weights#126");
    auto conv26 = om.conv(eltwise7, weights26, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_13/expand/Relu6#218");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (576);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4979446607176214e-05},{-inf},{inf}}, "expanded_conv_13/expand/Relu6#125_bias#127");
    auto bias_c26 = om.bias(conv26, biasWeights26, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData13 = mv::utils::generateSequence<int64_t> (3*3*576*1);
    auto d_weights13 = om.constantInt(d_weightsData13,{3,3,576,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0029497891664505005},{-0.38477009534835815},{0.3674261271953583}}, "expanded_conv_13/depthwise/Relu6#128_weights#129");
    auto depthConv13 = om.depthwiseConv(bias_c26, d_weights13, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_13/depthwise/Relu6#219");

    std::vector<int64_t> biasd_WeightsData13 = mv::utils::generateSequence<int64_t> (576);
    auto biasdWeights13 = om.constantInt(biasd_WeightsData13,{576}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1567800356715452e-05},{-inf},{inf}}, "expanded_conv_13/depthwise/Relu6#128_bias#130");
    auto bias_cd13 = om.bias(depthConv13, biasdWeights13, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*576*160);
    auto weights27 = om.constantInt(weightsData27,{1,1,576,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.003236441407352686},{-0.4176654815673828},{0.4076271057128906}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#131_weights#132");
    auto conv27 = om.conv(bias_cd13, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#220");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2691927622654475e-05},{-inf},{inf}}, "expanded_conv_13/project/BatchNorm/FusedBatchNorm/BiasAdd#131_bias#133");
    auto bias_c27 = om.bias(conv27, biasWeights27, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights28 = om.constantInt(weightsData28,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0033436305820941925},{-0.4370753765106201},{0.4155504107475281}}, "expanded_conv_14/expand/Relu6#134_weights#135");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_14/expand/Relu6#221");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.622455394885037e-05},{-inf},{inf}}, "expanded_conv_14/expand/Relu6#134_bias#136");
    auto bias_c28 = om.bias(conv28, biasWeights28, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData14 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights14 = om.constantInt(d_weightsData14,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.0032607857137918472},{-0.3759268522262573},{0.45557352900505066}}, "expanded_conv_14/depthwise/Relu6#137_weights#138");
    auto depthConv14 = om.depthwiseConv(bias_c28, d_weights14, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_14/depthwise/Relu6#222");

    std::vector<int64_t> biasd_WeightsData14 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights14 = om.constantInt(biasd_WeightsData14,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2787395462510176e-05},{-inf},{inf}}, "expanded_conv_14/depthwise/Relu6#137_bias#139");
    auto bias_cd14 = om.bias(depthConv14, biasdWeights14, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (1*1*960*160);
    auto weights29 = om.constantInt(weightsData29,{1,1,960,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003667778568342328},{-0.4963448643684387},{0.4389386773109436}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#140_weights#141");
    auto conv29 = om.conv(bias_cd14, weights29, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#223");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4383445886778645e-05},{-inf},{inf}}, "expanded_conv_14/project/BatchNorm/FusedBatchNorm/BiasAdd#140_bias#142");
    auto bias_c29 = om.bias(conv29, biasWeights29, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise8 = om.add({bias_c27,bias_c29}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_14/add/Add#224");

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights30 = om.constantInt(weightsData30,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.00334560452029109},{-0.4360058009624481},{0.41712334752082825}}, "expanded_conv_15/expand/Relu6#144_weights#145");
    auto conv30 = om.conv(eltwise8, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_15/expand/Relu6#225");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.624003536766395e-05},{-inf},{inf}}, "expanded_conv_15/expand/Relu6#144_bias#146");
    auto bias_c30 = om.bias(conv30, biasWeights30, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData15 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights15 = om.constantInt(d_weightsData15,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.00282974261790514},{-0.37195342779159546},{0.3496309220790863}}, "expanded_conv_15/depthwise/Relu6#147_weights#148");
    auto depthConv15 = om.depthwiseConv(bias_c30, d_weights15, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_15/depthwise/Relu6#226");

    std::vector<int64_t> biasd_WeightsData15 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights15 = om.constantInt(biasd_WeightsData15,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1097029528173152e-05},{-inf},{inf}}, "expanded_conv_15/depthwise/Relu6#147_bias#149");
    auto bias_cd15 = om.bias(depthConv15, biasdWeights15, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*960*160);
    auto weights31 = om.constantInt(weightsData31,{1,1,960,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0033787961583584547},{-0.4247035086154938},{0.43688949942588806}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#150_weights#151");
    auto conv31 = om.conv(bias_cd15, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#227");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.325018092757091e-05},{-inf},{inf}}, "expanded_conv_15/project/BatchNorm/FusedBatchNorm/BiasAdd#150_bias#152");
    auto bias_c31 = om.bias(conv31, biasWeights31, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto eltwise9 = om.add({eltwise8,bias_c31}, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_15/add/Add#228");

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (1*1*160*960);
    auto weights32 = om.constantInt(weightsData32,{1,1,160,960}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0036992442328482866},{-0.49393245577812195},{0.449374794960022}}, "expanded_conv_16/expand/Relu6#154_weights#155");
    auto conv32 = om.conv(eltwise9, weights32, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_16/expand/Relu6#229");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (960);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.9013679522904567e-05},{-inf},{inf}}, "expanded_conv_16/expand/Relu6#154_bias#156");
    auto bias_c32 = om.bias(conv32, biasWeights32, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> d_weightsData16 = mv::utils::generateSequence<int64_t> (3*3*960*1);
    auto d_weights16 = om.constantInt(d_weightsData16,{3,3,960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.002788091078400612},{-0.3722240626811981},{0.33873915672302246}}, "expanded_conv_16/depthwise/Relu6#157_weights#158");
    auto depthConv16 = om.depthwiseConv(bias_c32, d_weights16, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "expanded_conv_16/depthwise/Relu6#230");

    std::vector<int64_t> biasd_WeightsData16 = mv::utils::generateSequence<int64_t> (960);
    auto biasdWeights16 = om.constantInt(biasd_WeightsData16,{960}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0933690646197647e-05},{-inf},{inf}}, "expanded_conv_16/depthwise/Relu6#157_bias#159");
    auto bias_cd16 = om.bias(depthConv16, biasdWeights16, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*960*320);
    auto weights33 = om.constantInt(weightsData33,{1,1,960,320}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0035930087324231863},{-0.4399093687534332},{0.47630783915519714}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#160_weights#161");
    auto conv33 = om.conv(bias_cd16, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#231");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (320);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{320}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4090230251895264e-05},{-inf},{inf}}, "expanded_conv_16/project/BatchNorm/FusedBatchNorm/BiasAdd#160_bias#162");
    auto bias_c33 = om.bias(conv33, biasWeights33, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*320*1280);
    auto weights34 = om.constantInt(weightsData34,{1,1,320,1280}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003499699058011174},{-0.44588136672973633},{0.4465419054031372}}, "Conv_1/Relu6#163_weights#164");
    auto conv34 = om.conv(bias_c33, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "Conv_1/Relu6#232");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (1280);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{1280}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.7448621040093713e-05},{-inf},{inf}}, "Conv_1/Relu6#163_bias#165");
    auto bias_c34 = om.bias(conv34, biasWeights34, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.averagePool(bias_c34, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "Logits/AvgPool/AvgPool#233");

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (1*1*1280*1024);
    auto weights35 = om.constantInt(weightsData35,{1,1,1280,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.003863171674311161},{-0.5084267854690552},{0.47668200731277466}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd#167_weights#168");
    auto conv35 = om.conv(pool0, weights35, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd#234");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5149693354032934e-05},{-inf},{inf}}, "Logits/Conv2d_1c_1x1/BiasAdd/Logits/Conv2d_1c_1x1/BiasAdd#167_bias#169");
    auto bias_c35 = om.bias(conv35, biasWeights35, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c35);

    std::string compDescPath = params.comp_descriptor_;
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
