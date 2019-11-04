#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.04871978983283043},{-6.9027419090271},{5.472084999084473}}, "MobilenetV2/Conv/Relu6#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038211600622162223},{-inf},{inf}}, "MobilenetV2/Conv/Relu6#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (3*3*48*1);
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,48,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{95},{0.4666302800178528},{-44.04890823364258},{74.47518157958984}}, "MobilenetV2/expanded_conv/depthwise/Relu6#3_weights#4");
    auto depthConv0 = om.depthwiseConv(bias_c0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv/depthwise/Relu6#172");

    std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.010979099199175835},{-inf},{inf}}, "MobilenetV2/expanded_conv/depthwise/Relu6#3_bias#5");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*48*24);
    auto weights1 = om.constantInt(weightsData1,{1,1,48,24}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{167},{0.04477492719888687},{-7.413511753082275},{3.959320068359375}}, "MobilenetV2/expanded_conv/project/add_fold#6_weights#7");
    auto conv1 = om.conv(bias_cd0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{126},{0.3486804664134979},{-43.933738708496094},{44.97977828979492}}, "MobilenetV2/expanded_conv/project/add_fold#173");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (24);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{24}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0010534859029576182},{-inf},{inf}}, "MobilenetV2/expanded_conv/project/add_fold#6_bias#8");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{126},{0.3486804664134979},{-43.933738708496094},{44.97977828979492}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*24*144);
    auto weights2 = om.constantInt(weightsData2,{1,1,24,144}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.003087183926254511},{-0.40302228927612305},{0.3811224102973938}}, "MobilenetV2/expanded_conv_1/expand/Relu6#9_weights#10");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_1/expand/Relu6#174");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (144);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0010764406761154532},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/expand/Relu6#9_bias#11");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData1 = mv::utils::generateSequence<int64_t> (3*3*144*1);
    auto d_weights1 = om.constantInt(d_weightsData1,{3,3,144,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{103},{0.15689566731452942},{-15.936485290527344},{23.91501235961914}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#12_weights#13");
    auto depthConv1 = om.depthwiseConv(bias_c2, d_weights1, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#175");

    std::vector<int64_t> biasd_WeightsData1 = mv::utils::generateSequence<int64_t> (144);
    auto biasdWeights1 = om.constantInt(biasd_WeightsData1,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0036915158852934837},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#12_bias#14");
    auto bias_cd1 = om.bias(depthConv1, biasdWeights1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*144*32);
    auto weights3 = om.constantInt(weightsData3,{1,1,144,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.021310776472091675},{-2.8283071517944336},{2.584630250930786}}, "MobilenetV2/expanded_conv_1/project/add_fold#15_weights#16");
    auto conv3 = om.conv(bias_cd1, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{123},{0.3854878842830658},{-47.415008544921875},{50.884403228759766}}, "MobilenetV2/expanded_conv_1/project/add_fold#176");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005014101043343544},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/project/add_fold#15_bias#17");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("UInt8"), {{123},{0.3854878842830658},{-47.415008544921875},{50.884403228759766}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights4 = om.constantInt(weightsData4,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002736554481089115},{-0.2840699851512909},{0.4110148251056671}}, "MobilenetV2/expanded_conv_2/expand/Relu6#18_weights#19");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_2/expand/Relu6#177");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0010549086146056652},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/expand/Relu6#18_bias#20");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData2 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights2 = om.constantInt(d_weightsData2,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.054162535816431046},{-6.150203227996826},{7.607080459594727}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#21_weights#22");
    auto depthConv2 = om.depthwiseConv(bias_c4, d_weights2, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#178");

    std::vector<int64_t> biasd_WeightsData2 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights2 = om.constantInt(biasd_WeightsData2,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0012743619736284018},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#21_bias#23");
    auto bias_cd2 = om.bias(depthConv2, biasdWeights2, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights5 = om.constantInt(weightsData5,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.02357718162238598},{-2.6467220783233643},{3.3418822288513184}}, "MobilenetV2/expanded_conv_2/project/add_fold#24_weights#25");
    auto conv5 = om.conv(bias_cd2, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{123},{0.3854878842830658},{-47.415008544921875},{50.884403228759766}}, "MobilenetV2/expanded_conv_2/project/add_fold#179");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005547351902350783},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/project/add_fold#24_bias#26");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("UInt8"), {{123},{0.3854878842830658},{-47.415008544921875},{50.884403228759766}});

    auto eltwise0 = om.add({bias_c5,bias_c3}, mv::DType("UInt8"), {{126},{0.3989315927028656},{-50.265380859375},{51.46217346191406}}, "MobilenetV2/expanded_conv_2/add#180");

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights6 = om.constantInt(weightsData6,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.003178063780069351},{-0.35494738817214966},{0.45228078961372375}}, "MobilenetV2/expanded_conv_3/expand/Relu6#28_weights#29");
    auto conv6 = om.conv(eltwise0, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_3/expand/Relu6#181");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0012678300263360143},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/expand/Relu6#28_bias#30");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData3 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights3 = om.constantInt(d_weightsData3,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.016847463324666023},{-2.1872148513793945},{2.092041015625}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#31_weights#32");
    auto depthConv3 = om.depthwiseConv(bias_c6, d_weights3, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#182");

    std::vector<int64_t> biasd_WeightsData3 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights3 = om.constantInt(biasd_WeightsData3,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00039639516035094857},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#31_bias#33");
    auto bias_cd3 = om.bias(depthConv3, biasdWeights3, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*192*48);
    auto weights7 = om.constantInt(weightsData7,{1,1,192,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.013815086334943771},{-1.802670955657959},{1.706360936164856}}, "MobilenetV2/expanded_conv_3/project/add_fold#34_weights#35");
    auto conv7 = om.conv(bias_cd3, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{135},{0.21764534711837769},{-29.382122039794922},{26.117443084716797}}, "MobilenetV2/expanded_conv_3/project/add_fold#183");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003250479348935187},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/project/add_fold#34_bias#36");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("UInt8"), {{135},{0.21764534711837769},{-29.382122039794922},{26.117443084716797}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights8 = om.constantInt(weightsData8,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{100},{0.0028932993300259113},{-0.2876748740673065},{0.44722312688827515}}, "MobilenetV2/expanded_conv_4/expand/Relu6#37_weights#38");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_4/expand/Relu6#184");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006297131185419858},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/expand/Relu6#37_bias#39");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData4 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights4 = om.constantInt(d_weightsData4,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{42},{0.20124445855617523},{-8.333209037780762},{42.78288269042969}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#40_weights#41");
    auto depthConv4 = om.depthwiseConv(bias_c8, d_weights4, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#185");

    std::vector<int64_t> biasd_WeightsData4 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights4 = om.constantInt(biasd_WeightsData4,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0047349752858281136},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#40_bias#42");
    auto bias_cd4 = om.bias(depthConv4, biasdWeights4, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (1*1*288*48);
    auto weights9 = om.constantInt(weightsData9,{1,1,288,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.013498425483703613},{-1.6002382040023804},{1.8283618688583374}}, "MobilenetV2/expanded_conv_4/project/add_fold#43_weights#44");
    auto conv9 = om.conv(bias_cd4, weights9, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{135},{0.21764534711837769},{-29.382122039794922},{26.117443084716797}}, "MobilenetV2/expanded_conv_4/project/add_fold#186");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003175973834004253},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/project/add_fold#43_bias#45");
    auto bias_c9 = om.bias(conv9, biasWeights9, mv::DType("UInt8"), {{135},{0.21764534711837769},{-29.382122039794922},{26.117443084716797}});

    auto eltwise1 = om.add({bias_c9,bias_c7}, mv::DType("UInt8"), {{121},{0.24245388805866241},{-29.33692169189453},{32.48882293701172}}, "MobilenetV2/expanded_conv_4/add#187");

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights10 = om.constantInt(weightsData10,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0016727788606658578},{-0.2040429711341858},{0.2208428531885147}}, "MobilenetV2/expanded_conv_5/expand/Relu6#47_weights#48");
    auto conv10 = om.conv(eltwise1, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_5/expand/Relu6#188");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000405571743613109},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/expand/Relu6#47_bias#49");
    auto bias_c10 = om.bias(conv10, biasWeights10, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData5 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights5 = om.constantInt(d_weightsData5,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.048299793154001236},{-5.421045303344727},{6.847102165222168}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#50_weights#51");
    auto depthConv5 = om.depthwiseConv(bias_c10, d_weights5, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#189");

    std::vector<int64_t> biasd_WeightsData5 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights5 = om.constantInt(biasd_WeightsData5,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.001136420527473092},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#50_bias#52");
    auto bias_cd5 = om.bias(depthConv5, biasdWeights5, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*288*48);
    auto weights11 = om.constantInt(weightsData11,{1,1,288,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.016102129593491554},{-1.8154624700546265},{2.2744784355163574}}, "MobilenetV2/expanded_conv_5/project/add_fold#53_weights#54");
    auto conv11 = om.conv(bias_cd5, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{121},{0.24245388805866241},{-29.33692169189453},{32.48882293701172}}, "MobilenetV2/expanded_conv_5/project/add_fold#190");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00037885858910158277},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/project/add_fold#53_bias#55");
    auto bias_c11 = om.bias(conv11, biasWeights11, mv::DType("UInt8"), {{121},{0.24245388805866241},{-29.33692169189453},{32.48882293701172}});

    auto eltwise2 = om.add({bias_c11,eltwise1}, mv::DType("UInt8"), {{129},{0.2784804701805115},{-35.923980712890625},{35.088539123535156}}, "MobilenetV2/expanded_conv_5/add#191");

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights12 = om.constantInt(weightsData12,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{118},{0.0022823605686426163},{-0.2673361897468567},{0.31238338351249695}}, "MobilenetV2/expanded_conv_6/expand/Relu6#57_weights#58");
    auto conv12 = om.conv(eltwise2, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_6/expand/Relu6#192");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006355928489938378},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/expand/Relu6#57_bias#59");
    auto bias_c12 = om.bias(conv12, biasWeights12, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData6 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights6 = om.constantInt(d_weightsData6,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{106},{0.032531071454286575},{-3.426931142807007},{4.83596134185791}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#60_weights#61");
    auto depthConv6 = om.depthwiseConv(bias_c12, d_weights6, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#193");

    std::vector<int64_t> biasd_WeightsData6 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights6 = om.constantInt(biasd_WeightsData6,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007654065848328173},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#60_bias#62");
    auto bias_cd6 = om.bias(depthConv6, biasdWeights6, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (1*1*288*88);
    auto weights13 = om.constantInt(weightsData13,{1,1,288,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{140},{0.010896427556872368},{-1.5156992673873901},{1.2519932985305786}}, "MobilenetV2/expanded_conv_6/project/add_fold#63_weights#64");
    auto conv13 = om.conv(bias_cd6, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{124},{0.17781981825828552},{-22.049657821655273},{23.294397354125977}}, "MobilenetV2/expanded_conv_6/project/add_fold#194");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00025637634098529816},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/project/add_fold#63_bias#65");
    auto bias_c13 = om.bias(conv13, biasWeights13, mv::DType("UInt8"), {{124},{0.17781981825828552},{-22.049657821655273},{23.294397354125977}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights14 = om.constantInt(weightsData14,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{150},{0.0016304106684401631},{-0.24213194847106934},{0.17199234664440155}}, "MobilenetV2/expanded_conv_7/expand/Relu6#66_weights#67");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_7/expand/Relu6#195");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00028991932049393654},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/expand/Relu6#66_bias#68");
    auto bias_c14 = om.bias(conv14, biasWeights14, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData7 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights7 = om.constantInt(d_weightsData7,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{154},{0.0981987714767456},{-15.065185546875},{9.877303123474121}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#69_weights#70");
    auto depthConv7 = om.depthwiseConv(bias_c14, d_weights7, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#196");

    std::vector<int64_t> biasd_WeightsData7 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights7 = om.constantInt(biasd_WeightsData7,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0023104676511138678},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#69_bias#71");
    auto bias_cd7 = om.bias(depthConv7, biasdWeights7, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights15 = om.constantInt(weightsData15,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{147},{0.015141614712774754},{-2.21028733253479},{1.6356828212738037}}, "MobilenetV2/expanded_conv_7/project/add_fold#72_weights#73");
    auto conv15 = om.conv(bias_cd7, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{124},{0.17781981825828552},{-22.049657821655273},{23.294397354125977}}, "MobilenetV2/expanded_conv_7/project/add_fold#197");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000356259144609794},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/project/add_fold#72_bias#74");
    auto bias_c15 = om.bias(conv15, biasWeights15, mv::DType("UInt8"), {{124},{0.17781981825828552},{-22.049657821655273},{23.294397354125977}});

    auto eltwise3 = om.add({bias_c15,bias_c13}, mv::DType("UInt8"), {{125},{0.1822538673877716},{-22.7817325592041},{23.693002700805664}}, "MobilenetV2/expanded_conv_7/add#198");

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights16 = om.constantInt(weightsData16,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.001458671409636736},{-0.19451169669628143},{0.17599084973335266}}, "MobilenetV2/expanded_conv_8/expand/Relu6#76_weights#77");
    auto conv16 = om.conv(eltwise3, weights16, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_8/expand/Relu6#199");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00026584850274957716},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/expand/Relu6#76_bias#78");
    auto bias_c16 = om.bias(conv16, biasWeights16, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData8 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights8 = om.constantInt(d_weightsData8,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.03744745999574661},{-5.00202751159668},{4.509627342224121}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#79_weights#80");
    auto depthConv8 = om.depthwiseConv(bias_c16, d_weights8, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#200");

    std::vector<int64_t> biasd_WeightsData8 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights8 = om.constantInt(biasd_WeightsData8,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008810816798359156},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#79_bias#81");
    auto bias_cd8 = om.bias(depthConv8, biasdWeights8, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights17 = om.constantInt(weightsData17,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.01288581546396017},{-1.5957942008972168},{1.677202820777893}}, "MobilenetV2/expanded_conv_8/project/add_fold#82_weights#83");
    auto conv17 = om.conv(bias_cd8, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{125},{0.1822538673877716},{-22.7817325592041},{23.693002700805664}}, "MobilenetV2/expanded_conv_8/project/add_fold#201");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003031835949514061},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/project/add_fold#82_bias#84");
    auto bias_c17 = om.bias(conv17, biasWeights17, mv::DType("UInt8"), {{125},{0.1822538673877716},{-22.7817325592041},{23.693002700805664}});

    auto eltwise4 = om.add({bias_c17,eltwise3}, mv::DType("UInt8"), {{123},{0.19228406250476837},{-23.650938034057617},{25.38149642944336}}, "MobilenetV2/expanded_conv_8/add#202");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights18 = om.constantInt(weightsData18,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0013497850159183145},{-0.15888012945652008},{0.183965265750885}}, "MobilenetV2/expanded_conv_9/expand/Relu6#86_weights#87");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_9/expand/Relu6#203");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002595421392470598},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/expand/Relu6#86_bias#88");
    auto bias_c18 = om.bias(conv18, biasWeights18, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData9 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights9 = om.constantInt(d_weightsData9,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{0.03613342344760895},{-5.188309669494629},{3.9895801544189453}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#89_weights#90");
    auto depthConv9 = om.depthwiseConv(bias_c18, d_weights9, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#204");

    std::vector<int64_t> biasd_WeightsData9 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights9 = om.constantInt(biasd_WeightsData9,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008501644479110837},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#89_bias#91");
    auto bias_cd9 = om.bias(depthConv9, biasdWeights9, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights19 = om.constantInt(weightsData19,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{141},{0.012727844528853893},{-1.7835627794265747},{1.4493098258972168}}, "MobilenetV2/expanded_conv_9/project/add_fold#92_weights#93");
    auto conv19 = om.conv(bias_cd9, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{123},{0.19228406250476837},{-23.650938034057617},{25.38149642944336}}, "MobilenetV2/expanded_conv_9/project/add_fold#205");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002994668029714376},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/project/add_fold#92_bias#94");
    auto bias_c19 = om.bias(conv19, biasWeights19, mv::DType("UInt8"), {{123},{0.19228406250476837},{-23.650938034057617},{25.38149642944336}});

    auto eltwise5 = om.add({bias_c19,eltwise4}, mv::DType("UInt8"), {{124},{0.19631874561309814},{-24.343524932861328},{25.717754364013672}}, "MobilenetV2/expanded_conv_9/add#206");

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights20 = om.constantInt(weightsData20,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{0.0018451373325660825},{-0.26635316014289856},{0.202311709523201}}, "MobilenetV2/expanded_conv_10/expand/Relu6#96_weights#97");
    auto conv20 = om.conv(eltwise5, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_10/expand/Relu6#207");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00036223503411747515},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/expand/Relu6#96_bias#98");
    auto bias_c20 = om.bias(conv20, biasWeights20, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData10 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights10 = om.constantInt(d_weightsData10,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{154},{0.0302275437861681},{-4.638733386993408},{3.0390625}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#99_weights#100");
    auto depthConv10 = om.depthwiseConv(bias_c20, d_weights10, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#208");

    std::vector<int64_t> biasd_WeightsData10 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights10 = om.constantInt(biasd_WeightsData10,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007112080347724259},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#99_bias#101");
    auto bias_cd10 = om.bias(depthConv10, biasdWeights10, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*528*136);
    auto weights21 = om.constantInt(weightsData21,{1,1,528,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.005562151782214642},{-0.6710951924324036},{0.7416913509368896}}, "MobilenetV2/expanded_conv_10/project/add_fold#102_weights#103");
    auto conv21 = om.conv(bias_cd10, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{136},{0.15925274789333344},{-21.65837287902832},{18.95107650756836}}, "MobilenetV2/expanded_conv_10/project/add_fold#209");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00013086895341984928},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/project/add_fold#102_bias#104");
    auto bias_c21 = om.bias(conv21, biasWeights21, mv::DType("UInt8"), {{136},{0.15925274789333344},{-21.65837287902832},{18.95107650756836}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights22 = om.constantInt(weightsData22,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.002318246988579631},{-0.3094905912876129},{0.2793441712856293}}, "MobilenetV2/expanded_conv_11/expand/Relu6#105_weights#106");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_11/expand/Relu6#210");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003691872116178274},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/expand/Relu6#105_bias#107");
    auto bias_c22 = om.bias(conv22, biasWeights22, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData11 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights11 = om.constantInt(d_weightsData11,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{161},{0.10311193764209747},{-16.469228744506836},{9.721202850341797}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#108_weights#109");
    auto depthConv11 = om.depthwiseConv(bias_c22, d_weights11, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#211");

    std::vector<int64_t> biasd_WeightsData11 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights11 = om.constantInt(biasd_WeightsData11,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.002426066668704152},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#108_bias#110");
    auto bias_cd11 = om.bias(depthConv11, biasdWeights11, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*816*136);
    auto weights23 = om.constantInt(weightsData23,{1,1,816,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.007108029909431934},{-0.8901153206825256},{0.915324330329895}}, "MobilenetV2/expanded_conv_11/project/add_fold#111_weights#112");
    auto conv23 = om.conv(bias_cd11, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{136},{0.15925274789333344},{-21.65837287902832},{18.95107650756836}}, "MobilenetV2/expanded_conv_11/project/add_fold#212");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00016724111628718674},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/project/add_fold#111_bias#113");
    auto bias_c23 = om.bias(conv23, biasWeights23, mv::DType("UInt8"), {{136},{0.15925274789333344},{-21.65837287902832},{18.95107650756836}});

    auto eltwise6 = om.add({bias_c23,bias_c21}, mv::DType("UInt8"), {{134},{0.1797730028629303},{-24.089582443237305},{21.752532958984375}}, "MobilenetV2/expanded_conv_11/add#213");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights24 = om.constantInt(weightsData24,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.0012751613976433873},{-0.15654702484607697},{0.16734395921230316}}, "MobilenetV2/expanded_conv_12/expand/Relu6#115_weights#116");
    auto conv24 = om.conv(eltwise6, weights24, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_12/expand/Relu6#214");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022923958022147417},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/expand/Relu6#115_bias#117");
    auto bias_c24 = om.bias(conv24, biasWeights24, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData12 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights12 = om.constantInt(d_weightsData12,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{94},{0.08174128085374832},{-7.640551567077637},{13.121733665466309}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#118_weights#119");
    auto depthConv12 = om.depthwiseConv(bias_c24, d_weights12, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#215");

    std::vector<int64_t> biasd_WeightsData12 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights12 = om.constantInt(biasd_WeightsData12,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00192324782256037},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#118_bias#120");
    auto bias_cd12 = om.bias(depthConv12, biasdWeights12, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*816*136);
    auto weights25 = om.constantInt(weightsData25,{1,1,816,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.020400067791342735},{-2.6093571186065674},{2.5722601413726807}}, "MobilenetV2/expanded_conv_12/project/add_fold#121_weights#122");
    auto conv25 = om.conv(bias_cd12, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{134},{0.1797730028629303},{-24.089582443237305},{21.752532958984375}}, "MobilenetV2/expanded_conv_12/project/add_fold#216");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00047998252557590604},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/project/add_fold#121_bias#123");
    auto bias_c25 = om.bias(conv25, biasWeights25, mv::DType("UInt8"), {{134},{0.1797730028629303},{-24.089582443237305},{21.752532958984375}});

    auto eltwise7 = om.add({bias_c25,eltwise6}, mv::DType("UInt8"), {{129},{0.21570342779159546},{-27.825742721557617},{27.178630828857422}}, "MobilenetV2/expanded_conv_12/add#217");

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights26 = om.constantInt(weightsData26,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0017686078790575266},{-0.21449504792690277},{0.23473136126995087}}, "MobilenetV2/expanded_conv_13/expand/Relu6#125_weights#126");
    auto conv26 = om.conv(eltwise7, weights26, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_13/expand/Relu6#218");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038149478496052325},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/expand/Relu6#125_bias#127");
    auto bias_c26 = om.bias(conv26, biasWeights26, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData13 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights13 = om.constantInt(d_weightsData13,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.012934424914419651},{-1.6939210891723633},{1.5914229154586792}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#128_weights#129");
    auto depthConv13 = om.depthwiseConv(bias_c26, d_weights13, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#219");

    std::vector<int64_t> biasd_WeightsData13 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights13 = om.constantInt(biasd_WeightsData13,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00030432731728069484},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#128_bias#130");
    auto bias_cd13 = om.bias(depthConv13, biasdWeights13, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*816*224);
    auto weights27 = om.constantInt(weightsData27,{1,1,816,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.011571649461984634},{-1.3166073560714722},{1.6225916147232056}}, "MobilenetV2/expanded_conv_13/project/add_fold#131_weights#132");
    auto conv27 = om.conv(bias_cd13, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{132},{0.1171712577342987},{-15.466606140136719},{14.412064552307129}}, "MobilenetV2/expanded_conv_13/project/add_fold#220");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00027226327802054584},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/project/add_fold#131_bias#133");
    auto bias_c27 = om.bias(conv27, biasWeights27, mv::DType("UInt8"), {{132},{0.1171712577342987},{-15.466606140136719},{14.412064552307129}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights28 = om.constantInt(weightsData28,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0020822309888899326},{-0.24870409071445465},{0.2801826000213623}}, "MobilenetV2/expanded_conv_14/expand/Relu6#134_weights#135");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_14/expand/Relu6#221");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00024397762899752706},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/expand/Relu6#134_bias#136");
    auto bias_c28 = om.bias(conv28, biasWeights28, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData14 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights14 = om.constantInt(d_weightsData14,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.05240687355399132},{-6.17596435546875},{7.13538122177124}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#137_weights#138");
    auto depthConv14 = om.depthwiseConv(bias_c28, d_weights14, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#222");

    std::vector<int64_t> biasd_WeightsData14 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights14 = om.constantInt(biasd_WeightsData14,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0012330538593232632},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#137_bias#139");
    auto bias_cd14 = om.bias(depthConv14, biasdWeights14, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (1*1*1344*224);
    auto weights29 = om.constantInt(weightsData29,{1,1,1344,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{136},{0.008338418789207935},{-1.1252025365829468},{0.9927559494972229}}, "MobilenetV2/expanded_conv_14/project/add_fold#140_weights#141");
    auto conv29 = om.conv(bias_cd14, weights29, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{132},{0.1171712577342987},{-15.466606140136719},{14.412064552307129}}, "MobilenetV2/expanded_conv_14/project/add_fold#223");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001961903035407886},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/project/add_fold#140_bias#142");
    auto bias_c29 = om.bias(conv29, biasWeights29, mv::DType("UInt8"), {{132},{0.1171712577342987},{-15.466606140136719},{14.412064552307129}});

    auto eltwise8 = om.add({bias_c29,bias_c27}, mv::DType("UInt8"), {{126},{0.18661798536777496},{-23.513866424560547},{24.073720932006836}}, "MobilenetV2/expanded_conv_14/add#224");

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights30 = om.constantInt(weightsData30,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{144},{0.001718836254440248},{-0.24598367512226105},{0.19060073792934418}}, "MobilenetV2/expanded_conv_15/expand/Relu6#144_weights#145");
    auto conv30 = om.conv(eltwise8, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_15/expand/Relu6#225");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00032076577190309763},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/expand/Relu6#144_bias#146");
    auto bias_c30 = om.bias(conv30, biasWeights30, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData15 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights15 = om.constantInt(d_weightsData15,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0694165825843811},{-8.479001998901367},{9.152810096740723}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#147_weights#148");
    auto depthConv15 = om.depthwiseConv(bias_c30, d_weights15, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#226");

    std::vector<int64_t> biasd_WeightsData15 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights15 = om.constantInt(biasd_WeightsData15,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0016332664526998997},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#147_bias#149");
    auto bias_cd15 = om.bias(depthConv15, biasdWeights15, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*1344*224);
    auto weights31 = om.constantInt(weightsData31,{1,1,1344,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.033401161432266235},{-3.829883098602295},{4.6540117263793945}}, "MobilenetV2/expanded_conv_15/project/add_fold#150_weights#151");
    auto conv31 = om.conv(bias_cd15, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{126},{0.18661798536777496},{-23.513866424560547},{24.073720932006836}}, "MobilenetV2/expanded_conv_15/project/add_fold#227");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007858784520067275},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/project/add_fold#150_bias#152");
    auto bias_c31 = om.bias(conv31, biasWeights31, mv::DType("UInt8"), {{126},{0.18661798536777496},{-23.513866424560547},{24.073720932006836}});

    auto eltwise9 = om.add({bias_c31,eltwise8}, mv::DType("UInt8"), {{115},{0.22534283995628357},{-25.914426803588867},{31.547996520996094}}, "MobilenetV2/expanded_conv_15/add#228");

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights32 = om.constantInt(weightsData32,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0027454663068056107},{-0.3331298828125},{0.364218533039093}}, "MobilenetV2/expanded_conv_16/expand/Relu6#154_weights#155");
    auto conv32 = om.conv(eltwise9, weights32, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_16/expand/Relu6#229");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000618671125266701},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/expand/Relu6#154_bias#156");
    auto bias_c32 = om.bias(conv32, biasWeights32, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData16 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights16 = om.constantInt(d_weightsData16,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{200},{0.1743541955947876},{-34.66250991821289},{9.623455047607422}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#157_weights#158");
    auto depthConv16 = om.depthwiseConv(bias_c32, d_weights16, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#230");

    std::vector<int64_t> biasd_WeightsData16 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights16 = om.constantInt(biasd_WeightsData16,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.004102288745343685},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#157_bias#159");
    auto bias_cd16 = om.bias(depthConv16, biasdWeights16, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*1344*448);
    auto weights33 = om.constantInt(weightsData33,{1,1,1344,448}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.006061397958546877},{-0.8624204397201538},{0.6771746277809143}}, "MobilenetV2/expanded_conv_16/project/add_fold#160_weights#161");
    auto conv33 = om.conv(bias_cd16, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{127},{0.08560898900032043},{-10.87234115600586},{10.957950592041016}}, "MobilenetV2/expanded_conv_16/project/add_fold#231");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (448);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{448}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001426154631190002},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/project/add_fold#160_bias#162");
    auto bias_c33 = om.bias(conv33, biasWeights33, mv::DType("UInt8"), {{127},{0.08560898900032043},{-10.87234115600586},{10.957950592041016}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*448*1792);
    auto weights34 = om.constantInt(weightsData34,{1,1,448,1792}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{139},{0.004873978905379772},{-0.6718925833702087},{0.5660980939865112}}, "MobilenetV2/Conv_1/Relu6#163_weights#164");
    auto conv34 = om.conv(bias_c33, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv_1/Relu6#232");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (1792);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{1792}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00041725640767253935},{-inf},{inf}}, "MobilenetV2/Conv_1/Relu6#163_bias#165");
    auto bias_c34 = om.bias(conv34, biasWeights34, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    auto pool0 = om.averagePool(bias_c34, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Logits/AvgPool#233");

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (1*1*1792*1001);
    auto weights35 = om.constantInt(weightsData35,{1,1,1792,1001}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{106},{0.0015488684875890613},{-0.16265293955802917},{0.2307596653699875}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#167_weights#168");
    auto conv35 = om.conv(pool0, weights35, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{70},{0.09544266015291214},{-6.680986404418945},{17.656892776489258}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#234");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (1001);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{1001}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{3.644251773948781e-05},{-inf},{inf}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#167_bias#169");
    auto bias_c35 = om.bias(conv35, biasWeights35, mv::DType("UInt8"), {{70},{0.09544266015291214},{-6.680986404418945},{17.656892776489258}});

    om.output(bias_c35);

    std::string compDescPath = "/home/tbartsok/Desktop/WORK/mcmCompiler/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
