//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("MobileNetV2");
    mv::OpModel& om = unit.model();
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{140},{0.035487376153469086},{-4.933721542358398},{4.080072402954102}}, "MobilenetV2/Conv/Relu6#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002783323870971799},{-inf},{inf}}, "MobilenetV2/Conv/Relu6#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (3*3*48*1);
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,48,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{95},{0.4666302800178528},{-44.04890823364258},{74.47518157958984}}, "MobilenetV2/expanded_conv/depthwise/Relu6#3_weights#4");
    auto depthConv0 = om.depthwiseConv(bias_c0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv/depthwise/Relu6#172");

    std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.010979099199175835},{-inf},{inf}}, "MobilenetV2/expanded_conv/depthwise/Relu6#3_bias#5");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*48*24);
    auto weights1 = om.constantInt(weightsData1,{1,1,48,24}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{170},{0.04807610064744949},{-8.12545108795166},{4.085878372192383}}, "MobilenetV2/expanded_conv/project/add_fold#6_weights#7");
    auto conv1 = om.conv(bias_cd0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{130},{0.39397189021110535},{-51.21634292602539},{49.24648666381836}}, "MobilenetV2/expanded_conv/project/add_fold#173");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (24);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{24}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0011311573907732964},{-inf},{inf}}, "MobilenetV2/expanded_conv/project/add_fold#6_bias#8");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{130},{0.39397189021110535},{-51.21634292602539},{49.24648666381836}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*24*144);
    auto weights2 = om.constantInt(weightsData2,{1,1,24,144}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0032281773164868355},{-0.41518232226371765},{0.4047746956348419}}, "MobilenetV2/expanded_conv_1/expand/Relu6#9_weights#10");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_1/expand/Relu6#174");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (144);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00127181108109653},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/expand/Relu6#9_bias#11");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData1 = mv::utils::generateSequence<int64_t> (3*3*144*1);
    auto d_weights1 = om.constantInt(d_weightsData1,{3,3,144,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{103},{0.15689566731452942},{-15.936485290527344},{23.91501235961914}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#12_weights#13");
    auto depthConv1 = om.depthwiseConv(bias_c2, d_weights1, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#175");

    std::vector<int64_t> biasd_WeightsData1 = mv::utils::generateSequence<int64_t> (144);
    auto biasdWeights1 = om.constantInt(biasd_WeightsData1,{144}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0036915158852934837},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/depthwise/Relu6#12_bias#14");
    auto bias_cd1 = om.bias(depthConv1, biasdWeights1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*144*32);
    auto weights3 = om.constantInt(weightsData3,{1,1,144,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{155},{0.019840337336063385},{-3.057819128036499},{1.9816265106201172}}, "MobilenetV2/expanded_conv_1/project/add_fold#15_weights#16");
    auto conv3 = om.conv(bias_cd1, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{126},{0.40382319688796997},{-50.8817253112793},{52.09319305419922}}, "MobilenetV2/expanded_conv_1/project/add_fold#176");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004668129258789122},{-inf},{inf}}, "MobilenetV2/expanded_conv_1/project/add_fold#15_bias#17");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("UInt8"), {{126},{0.40382319688796997},{-50.8817253112793},{52.09319305419922}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights4 = om.constantInt(weightsData4,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.0027541988529264927},{-0.30907925963401794},{0.3904872536659241}}, "MobilenetV2/expanded_conv_2/expand/Relu6#18_weights#19");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_2/expand/Relu6#177");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0011122094001621008},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/expand/Relu6#18_bias#20");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData2 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights2 = om.constantInt(d_weightsData2,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{110},{0.05635802447795868},{-6.1420159339904785},{8.172922134399414}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#21_weights#22");
    auto depthConv2 = om.depthwiseConv(bias_c4, d_weights2, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#178");

    std::vector<int64_t> biasd_WeightsData2 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights2 = om.constantInt(biasd_WeightsData2,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0013260184787213802},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/depthwise/Relu6#21_bias#23");
    auto bias_cd2 = om.bias(depthConv2, biasdWeights2, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights5 = om.constantInt(weightsData5,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{117},{0.023070678114891052},{-2.676447629928589},{3.183504819869995}}, "MobilenetV2/expanded_conv_2/project/add_fold#24_weights#25");
    auto conv5 = om.conv(bias_cd2, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{126},{0.40382319688796997},{-50.8817253112793},{52.09319305419922}}, "MobilenetV2/expanded_conv_2/project/add_fold#179");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005428179283626378},{-inf},{inf}}, "MobilenetV2/expanded_conv_2/project/add_fold#24_bias#26");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("UInt8"), {{126},{0.40382319688796997},{-50.8817253112793},{52.09319305419922}});

    auto eltwise0 = om.eltwise({bias_c5,bias_c3}, "Add", mv::DType("UInt8"), {{126},{0.41732993721961975},{-52.58357238769531},{53.83556365966797}}, "MobilenetV2/expanded_conv_2/add#180");

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*32*192);
    auto weights6 = om.constantInt(weightsData6,{1,1,32,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.003304207930341363},{-0.3663899600505829},{0.4728788733482361}}, "MobilenetV2/expanded_conv_3/expand/Relu6#28_weights#29");
    auto conv6 = om.conv(eltwise0, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_3/expand/Relu6#181");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0013789449585601687},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/expand/Relu6#28_bias#30");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData3 = mv::utils::generateSequence<int64_t> (3*3*192*1);
    auto d_weights3 = om.constantInt(d_weightsData3,{3,3,192,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0173240527510643},{-2.3076462745666504},{2.092663288116455}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#31_weights#32");
    auto depthConv3 = om.depthwiseConv(bias_c6, d_weights3, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#182");

    std::vector<int64_t> biasd_WeightsData3 = mv::utils::generateSequence<int64_t> (192);
    auto biasdWeights3 = om.constantInt(biasd_WeightsData3,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004076085751876235},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/depthwise/Relu6#31_bias#33");
    auto bias_cd3 = om.bias(depthConv3, biasdWeights3, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*192*48);
    auto weights7 = om.constantInt(weightsData7,{1,1,192,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.014738907106220722},{-1.9012141227722168},{1.84246826171875}}, "MobilenetV2/expanded_conv_3/project/add_fold#34_weights#35");
    auto conv7 = om.conv(bias_cd3, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{138},{0.22786590456962585},{-31.445493698120117},{26.660310745239258}}, "MobilenetV2/expanded_conv_3/project/add_fold#183");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003467840433586389},{-inf},{inf}}, "MobilenetV2/expanded_conv_3/project/add_fold#34_bias#36");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("UInt8"), {{138},{0.22786590456962585},{-31.445493698120117},{26.660310745239258}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights8 = om.constantInt(weightsData8,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{94},{0.002917465753853321},{-0.27143457531929016},{0.46960175037384033}}, "MobilenetV2/expanded_conv_4/expand/Relu6#37_weights#38");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_4/expand/Relu6#184");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006647909758612514},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/expand/Relu6#37_bias#39");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData4 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights4 = om.constantInt(d_weightsData4,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{34},{0.19805563986301422},{-6.539368629455566},{43.76676559448242}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#40_weights#41");
    auto depthConv4 = om.depthwiseConv(bias_c8, d_weights4, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#185");

    std::vector<int64_t> biasd_WeightsData4 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights4 = om.constantInt(biasd_WeightsData4,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.004659947473555803},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/depthwise/Relu6#40_bias#42");
    auto bias_cd4 = om.bias(depthConv4, biasdWeights4, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (1*1*288*48);
    auto weights9 = om.constantInt(weightsData9,{1,1,288,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.013875649310648441},{-1.7192857265472412},{1.8051291704177856}}, "MobilenetV2/expanded_conv_4/project/add_fold#43_weights#44");
    auto conv9 = om.conv(bias_cd4, weights9, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{138},{0.22786590456962585},{-31.445493698120117},{26.660310745239258}}, "MobilenetV2/expanded_conv_4/project/add_fold#186");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003264728875365108},{-inf},{inf}}, "MobilenetV2/expanded_conv_4/project/add_fold#43_bias#45");
    auto bias_c9 = om.bias(conv9, biasWeights9, mv::DType("UInt8"), {{138},{0.22786590456962585},{-31.445493698120117},{26.660310745239258}});

    auto eltwise1 = om.eltwise({bias_c9,bias_c7}, "Add", mv::DType("UInt8"), {{122},{0.25491422414779663},{-31.099533081054688},{33.90359115600586}}, "MobilenetV2/expanded_conv_4/add#187");

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights10 = om.constantInt(weightsData10,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0016598537331447005},{-0.2136755734682083},{0.20792728662490845}}, "MobilenetV2/expanded_conv_5/expand/Relu6#47_weights#48");
    auto conv10 = om.conv(eltwise1, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_5/expand/Relu6#188");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000423120305640623},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/expand/Relu6#47_bias#49");
    auto bias_c10 = om.bias(conv10, biasWeights10, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData5 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights5 = om.constantInt(d_weightsData5,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{111},{0.05013015866279602},{-5.497150897979736},{7.2359089851379395}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#50_weights#51");
    auto depthConv5 = om.depthwiseConv(bias_c10, d_weights5, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#189");

    std::vector<int64_t> biasd_WeightsData5 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights5 = om.constantInt(biasd_WeightsData5,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0011794862803071737},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/depthwise/Relu6#50_bias#52");
    auto bias_cd5 = om.bias(depthConv5, biasdWeights5, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*288*48);
    auto weights11 = om.constantInt(weightsData11,{1,1,288,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.01638936810195446},{-1.8521205186843872},{2.310779094696045}}, "MobilenetV2/expanded_conv_5/project/add_fold#53_weights#54");
    auto conv11 = om.conv(bias_cd5, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{122},{0.25491422414779663},{-31.099533081054688},{33.90359115600586}}, "MobilenetV2/expanded_conv_5/project/add_fold#190");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038561687688343227},{-inf},{inf}}, "MobilenetV2/expanded_conv_5/project/add_fold#53_bias#55");
    auto bias_c11 = om.bias(conv11, biasWeights11, mv::DType("UInt8"), {{122},{0.25491422414779663},{-31.099533081054688},{33.90359115600586}});

    auto eltwise2 = om.eltwise({bias_c11,eltwise1}, "Add", mv::DType("UInt8"), {{130},{0.29750749468803406},{-38.67597579956055},{37.18843460083008}}, "MobilenetV2/expanded_conv_5/add#191");

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*48*288);
    auto weights12 = om.constantInt(weightsData12,{1,1,48,288}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.002265834016725421},{-0.2694503664970398},{0.3060714602470398}}, "MobilenetV2/expanded_conv_6/expand/Relu6#57_weights#58");
    auto conv12 = om.conv(eltwise2, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_6/expand/Relu6#192");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (288);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006741025717929006},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/expand/Relu6#57_bias#59");
    auto bias_c12 = om.bias(conv12, biasWeights12, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData6 = mv::utils::generateSequence<int64_t> (3*3*288*1);
    auto d_weights6 = om.constantInt(d_weightsData6,{3,3,288,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{108},{0.03464187681674957},{-3.7045278549194336},{5.094509124755859}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#60_weights#61");
    auto depthConv6 = om.depthwiseConv(bias_c12, d_weights6, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#193");

    std::vector<int64_t> biasd_WeightsData6 = mv::utils::generateSequence<int64_t> (288);
    auto biasdWeights6 = om.constantInt(biasd_WeightsData6,{288}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008150706416927278},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/depthwise/Relu6#60_bias#62");
    auto bias_cd6 = om.bias(depthConv6, biasdWeights6, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (1*1*288*88);
    auto weights13 = om.constantInt(weightsData13,{1,1,288,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{144},{0.011429100297391415},{-1.63445246219635},{1.2685389518737793}}, "MobilenetV2/expanded_conv_6/project/add_fold#63_weights#64");
    auto conv13 = om.conv(bias_cd6, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{123},{0.1842048019170761},{-22.657190322875977},{24.315034866333008}}, "MobilenetV2/expanded_conv_6/project/add_fold#194");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002689093234948814},{-inf},{inf}}, "MobilenetV2/expanded_conv_6/project/add_fold#63_bias#65");
    auto bias_c13 = om.bias(conv13, biasWeights13, mv::DType("UInt8"), {{123},{0.1842048019170761},{-22.657190322875977},{24.315034866333008}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights14 = om.constantInt(weightsData14,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.001447562943212688},{-0.18097859621047974},{0.1867024004459381}}, "MobilenetV2/expanded_conv_7/expand/Relu6#66_weights#67");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_7/expand/Relu6#195");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00026664804317988455},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/expand/Relu6#66_bias#68");
    auto bias_c14 = om.bias(conv14, biasWeights14, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData7 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights7 = om.constantInt(d_weightsData7,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{168},{0.10434550046920776},{-17.375530242919922},{9.128226280212402}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#69_weights#70");
    auto depthConv7 = om.depthwiseConv(bias_c14, d_weights7, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#196");

    std::vector<int64_t> biasd_WeightsData7 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights7 = om.constantInt(biasd_WeightsData7,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0024550906382501125},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/depthwise/Relu6#69_bias#71");
    auto bias_cd7 = om.bias(depthConv7, biasdWeights7, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights15 = om.constantInt(weightsData15,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{144},{0.014014680869877338},{-2.0062413215637207},{1.5534876585006714}}, "MobilenetV2/expanded_conv_7/project/add_fold#72_weights#73");
    auto conv15 = om.conv(bias_cd7, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{123},{0.1842048019170761},{-22.657190322875977},{24.315034866333008}}, "MobilenetV2/expanded_conv_7/project/add_fold#197");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003297440998721868},{-inf},{inf}}, "MobilenetV2/expanded_conv_7/project/add_fold#72_bias#74");
    auto bias_c15 = om.bias(conv15, biasWeights15, mv::DType("UInt8"), {{123},{0.1842048019170761},{-22.657190322875977},{24.315034866333008}});

    auto eltwise3 = om.eltwise({bias_c15,bias_c13}, "Add", mv::DType("UInt8"), {{124},{0.18766838312149048},{-23.2708797454834},{24.584556579589844}}, "MobilenetV2/expanded_conv_7/add#198");

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights16 = om.constantInt(weightsData16,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.0014362800866365433},{-0.2043149471282959},{0.1605001837015152}}, "MobilenetV2/expanded_conv_8/expand/Relu6#76_weights#77");
    auto conv16 = om.conv(eltwise3, weights16, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_8/expand/Relu6#199");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00026954433997161686},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/expand/Relu6#76_bias#78");
    auto bias_c16 = om.bias(conv16, biasWeights16, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData8 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights8 = om.constantInt(d_weightsData8,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.03615300729870796},{-4.805455207824707},{4.377408981323242}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#79_weights#80");
    auto depthConv8 = om.depthwiseConv(bias_c16, d_weights8, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#200");

    std::vector<int64_t> biasd_WeightsData8 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights8 = om.constantInt(biasd_WeightsData8,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008506252197548747},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/depthwise/Relu6#79_bias#81");
    auto bias_cd8 = om.bias(depthConv8, biasdWeights8, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights17 = om.constantInt(weightsData17,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.01368709746748209},{-1.6834827661514282},{1.7930399179458618}}, "MobilenetV2/expanded_conv_8/project/add_fold#82_weights#83");
    auto conv17 = om.conv(bias_cd8, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{124},{0.18766838312149048},{-23.2708797454834},{24.584556579589844}}, "MobilenetV2/expanded_conv_8/project/add_fold#201");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00032203656155616045},{-inf},{inf}}, "MobilenetV2/expanded_conv_8/project/add_fold#82_bias#84");
    auto bias_c17 = om.bias(conv17, biasWeights17, mv::DType("UInt8"), {{124},{0.18766838312149048},{-23.2708797454834},{24.584556579589844}});

    auto eltwise4 = om.eltwise({bias_c17,eltwise3}, "Add", mv::DType("UInt8"), {{124},{0.20268955826759338},{-25.13350486755371},{26.552331924438477}}, "MobilenetV2/expanded_conv_8/add#202");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights18 = om.constantInt(weightsData18,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0013557319762185216},{-0.1599821001291275},{0.18437382578849792}}, "MobilenetV2/expanded_conv_9/expand/Relu6#86_weights#87");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_9/expand/Relu6#203");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000274792721029371},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/expand/Relu6#86_bias#88");
    auto bias_c18 = om.bias(conv18, biasWeights18, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData9 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights9 = om.constantInt(d_weightsData9,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{142},{0.03765737637877464},{-5.298866271972656},{4.266107082366943}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#89_weights#90");
    auto depthConv9 = om.depthwiseConv(bias_c18, d_weights9, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#204");

    std::vector<int64_t> biasd_WeightsData9 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights9 = om.constantInt(biasd_WeightsData9,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008860206580720842},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/depthwise/Relu6#89_bias#91");
    auto bias_cd9 = om.bias(depthConv9, biasdWeights9, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (1*1*528*88);
    auto weights19 = om.constantInt(weightsData19,{1,1,528,88}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{141},{0.013033668510615826},{-1.8250097036361694},{1.4855420589447021}}, "MobilenetV2/expanded_conv_9/project/add_fold#92_weights#93");
    auto conv19 = om.conv(bias_cd9, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{124},{0.20268955826759338},{-25.13350486755371},{26.552331924438477}}, "MobilenetV2/expanded_conv_9/project/add_fold#205");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (88);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{88}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003066623758058995},{-inf},{inf}}, "MobilenetV2/expanded_conv_9/project/add_fold#92_bias#94");
    auto bias_c19 = om.bias(conv19, biasWeights19, mv::DType("UInt8"), {{124},{0.20268955826759338},{-25.13350486755371},{26.552331924438477}});

    auto eltwise5 = om.eltwise({bias_c19,eltwise4}, "Add", mv::DType("UInt8"), {{125},{0.2143997997045517},{-26.79997444152832},{27.87197494506836}}, "MobilenetV2/expanded_conv_9/add#206");

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*88*528);
    auto weights20 = om.constantInt(weightsData20,{1,1,88,528}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{0.0019500608323141932},{-0.2807634472846985},{0.21455200016498566}}, "MobilenetV2/expanded_conv_10/expand/Relu6#96_weights#97");
    auto conv20 = om.conv(eltwise5, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_10/expand/Relu6#207");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (528);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004180926480330527},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/expand/Relu6#96_bias#98");
    auto bias_c20 = om.bias(conv20, biasWeights20, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData10 = mv::utils::generateSequence<int64_t> (3*3*528*1);
    auto d_weights10 = om.constantInt(d_weightsData10,{3,3,528,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{159},{0.03171578422188759},{-5.016012191772461},{3.039796829223633}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#99_weights#100");
    auto depthConv10 = om.depthwiseConv(bias_c20, d_weights10, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#208");

    std::vector<int64_t> biasd_WeightsData10 = mv::utils::generateSequence<int64_t> (528);
    auto biasdWeights10 = om.constantInt(biasd_WeightsData10,{528}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007462240755558014},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/depthwise/Relu6#99_bias#101");
    auto bias_cd10 = om.bias(depthConv10, biasdWeights10, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*528*136);
    auto weights21 = om.constantInt(weightsData21,{1,1,528,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.006011901423335075},{-0.7453112006187439},{0.781711757183075}}, "MobilenetV2/expanded_conv_10/project/add_fold#102_weights#103");
    auto conv21 = om.conv(bias_cd10, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{137},{0.16858015954494476},{-23.095481872558594},{19.892457962036133}}, "MobilenetV2/expanded_conv_10/project/add_fold#209");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00014145088789518923},{-inf},{inf}}, "MobilenetV2/expanded_conv_10/project/add_fold#102_bias#104");
    auto bias_c21 = om.bias(conv21, biasWeights21, mv::DType("UInt8"), {{137},{0.16858015954494476},{-23.095481872558594},{19.892457962036133}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights22 = om.constantInt(weightsData22,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.002328268252313137},{-0.3114250898361206},{0.27995505928993225}}, "MobilenetV2/expanded_conv_11/expand/Relu6#105_weights#106");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_11/expand/Relu6#210");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003924998454749584},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/expand/Relu6#105_bias#107");
    auto bias_c22 = om.bias(conv22, biasWeights22, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData11 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights11 = om.constantInt(d_weightsData11,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{161},{0.09341112524271011},{-14.929360389709473},{8.797064781188965}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#108_weights#109");
    auto depthConv11 = om.depthwiseConv(bias_c22, d_weights11, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#211");

    std::vector<int64_t> biasd_WeightsData11 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights11 = om.constantInt(biasd_WeightsData11,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.002197821391746402},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/depthwise/Relu6#108_bias#110");
    auto bias_cd11 = om.bias(depthConv11, biasdWeights11, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*816*136);
    auto weights23 = om.constantInt(weightsData23,{1,1,816,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0069654639810323715},{-0.8625586032867432},{0.9066691994667053}}, "MobilenetV2/expanded_conv_11/project/add_fold#111_weights#112");
    auto conv23 = om.conv(bias_cd11, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{137},{0.16858015954494476},{-23.095481872558594},{19.892457962036133}}, "MobilenetV2/expanded_conv_11/project/add_fold#212");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001638867543078959},{-inf},{inf}}, "MobilenetV2/expanded_conv_11/project/add_fold#111_bias#113");
    auto bias_c23 = om.bias(conv23, biasWeights23, mv::DType("UInt8"), {{137},{0.16858015954494476},{-23.095481872558594},{19.892457962036133}});

    auto eltwise6 = om.eltwise({bias_c23,bias_c21}, "Add", mv::DType("UInt8"), {{130},{0.18281149864196777},{-23.76549530029297},{22.851438522338867}}, "MobilenetV2/expanded_conv_11/add#213");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights24 = om.constantInt(weightsData24,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0013235065853223205},{-0.1600651741027832},{0.17610549926757812}}, "MobilenetV2/expanded_conv_12/expand/Relu6#115_weights#116");
    auto conv24 = om.conv(eltwise6, weights24, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_12/expand/Relu6#214");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00024195223522838205},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/expand/Relu6#115_bias#117");
    auto bias_c24 = om.bias(conv24, biasWeights24, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData12 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights12 = om.constantInt(d_weightsData12,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{85},{0.08436132222414017},{-7.045048713684082},{14.382726669311523}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#118_weights#119");
    auto depthConv12 = om.depthwiseConv(bias_c24, d_weights12, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#215");

    std::vector<int64_t> biasd_WeightsData12 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights12 = om.constantInt(biasd_WeightsData12,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0019848933443427086},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/depthwise/Relu6#118_bias#120");
    auto bias_cd12 = om.bias(depthConv12, biasdWeights12, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*816*136);
    auto weights25 = om.constantInt(weightsData25,{1,1,816,136}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.021184345707297325},{-2.711790084838867},{2.6690337657928467}}, "MobilenetV2/expanded_conv_12/project/add_fold#121_weights#122");
    auto conv25 = om.conv(bias_cd12, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{130},{0.18281149864196777},{-23.76549530029297},{22.851438522338867}}, "MobilenetV2/expanded_conv_12/project/add_fold#216");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (136);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{136}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004984354018233716},{-inf},{inf}}, "MobilenetV2/expanded_conv_12/project/add_fold#121_bias#123");
    auto bias_c25 = om.bias(conv25, biasWeights25, mv::DType("UInt8"), {{130},{0.18281149864196777},{-23.76549530029297},{22.851438522338867}});

    auto eltwise7 = om.eltwise({bias_c25,eltwise6}, "Add", mv::DType("UInt8"), {{124},{0.23920400440692902},{-29.661296844482422},{31.335725784301758}}, "MobilenetV2/expanded_conv_12/add#217");

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (1*1*136*816);
    auto weights26 = om.constantInt(weightsData26,{1,1,136,816}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.001924174022860825},{-0.23847021162509918},{0.25026997923851013}}, "MobilenetV2/expanded_conv_13/expand/Relu6#125_weights#126");
    auto conv26 = om.conv(eltwise7, weights26, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_13/expand/Relu6#218");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (816);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00046027012285776436},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/expand/Relu6#125_bias#127");
    auto bias_c26 = om.bias(conv26, biasWeights26, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData13 = mv::utils::generateSequence<int64_t> (3*3*816*1);
    auto d_weights13 = om.constantInt(d_weightsData13,{3,3,816,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.013492844998836517},{-1.79688560962677},{1.6302969455718994}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#128_weights#129");
    auto depthConv13 = om.depthwiseConv(bias_c26, d_weights13, {2, 2}, {0, 1, 0, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#219");

    std::vector<int64_t> biasd_WeightsData13 = mv::utils::generateSequence<int64_t> (816);
    auto biasdWeights13 = om.constantInt(biasd_WeightsData13,{816}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000317466096021235},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/depthwise/Relu6#128_bias#130");
    auto bias_cd13 = om.bias(depthConv13, biasdWeights13, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*816*224);
    auto weights27 = om.constantInt(weightsData27,{1,1,816,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.010889791883528233},{-1.2310699224472046},{1.5349372625350952}}, "MobilenetV2/expanded_conv_13/project/add_fold#131_weights#132");
    auto conv27 = om.conv(bias_cd13, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{131},{0.13069577515125275},{-17.121145248413086},{16.206275939941406}}, "MobilenetV2/expanded_conv_13/project/add_fold#220");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00025622022803872824},{-inf},{inf}}, "MobilenetV2/expanded_conv_13/project/add_fold#131_bias#133");
    auto bias_c27 = om.bias(conv27, biasWeights27, mv::DType("UInt8"), {{131},{0.13069577515125275},{-17.121145248413086},{16.206275939941406}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights28 = om.constantInt(weightsData28,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.0021485236939042807},{-0.263288676738739},{0.282436341047287}}, "MobilenetV2/expanded_conv_14/expand/Relu6#134_weights#135");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_14/expand/Relu6#221");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002808029530569911},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/expand/Relu6#134_bias#136");
    auto bias_c28 = om.bias(conv28, biasWeights28, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData14 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights14 = om.constantInt(d_weightsData14,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.055074505507946014},{-6.087097644805908},{7.901826858520508}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#137_weights#138");
    auto depthConv14 = om.depthwiseConv(bias_c28, d_weights14, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#222");

    std::vector<int64_t> biasd_WeightsData14 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights14 = om.constantInt(biasd_WeightsData14,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.001295819180086255},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/depthwise/Relu6#137_bias#139");
    auto bias_cd14 = om.bias(depthConv14, biasdWeights14, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (1*1*1344*224);
    auto weights29 = om.constantInt(weightsData29,{1,1,1344,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{147},{0.008314176462590694},{-1.214598298072815},{0.8972024321556091}}, "MobilenetV2/expanded_conv_14/project/add_fold#140_weights#141");
    auto conv29 = om.conv(bias_cd14, weights29, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{131},{0.13069577515125275},{-17.121145248413086},{16.206275939941406}}, "MobilenetV2/expanded_conv_14/project/add_fold#223");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00019561989756766707},{-inf},{inf}}, "MobilenetV2/expanded_conv_14/project/add_fold#140_bias#142");
    auto bias_c29 = om.bias(conv29, biasWeights29, mv::DType("UInt8"), {{131},{0.13069577515125275},{-17.121145248413086},{16.206275939941406}});

    auto eltwise8 = om.eltwise({bias_c29,bias_c27}, "Add", mv::DType("UInt8"), {{132},{0.16261771321296692},{-21.465538024902344},{20.001977920532227}}, "MobilenetV2/expanded_conv_14/add#224");

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights30 = om.constantInt(weightsData30,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{152},{0.0017869345610961318},{-0.26938650012016296},{0.18449488282203674}}, "MobilenetV2/expanded_conv_15/expand/Relu6#144_weights#145");
    auto conv30 = om.conv(eltwise8, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_15/expand/Relu6#225");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00029058719519525766},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/expand/Relu6#144_bias#146");
    auto bias_c30 = om.bias(conv30, biasWeights30, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData15 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights15 = om.constantInt(d_weightsData15,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.0726834312081337},{-8.224613189697266},{10.236978530883789}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#147_weights#148");
    auto depthConv15 = om.depthwiseConv(bias_c30, d_weights15, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#226");

    std::vector<int64_t> biasd_WeightsData15 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights15 = om.constantInt(biasd_WeightsData15,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0017101304838433862},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/depthwise/Relu6#147_bias#149");
    auto bias_cd15 = om.bias(depthConv15, biasdWeights15, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*1344*224);
    auto weights31 = om.constantInt(weightsData31,{1,1,1344,224}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.03132287412881851},{-3.571418285369873},{4.384591579437256}}, "MobilenetV2/expanded_conv_15/project/add_fold#150_weights#151");
    auto conv31 = om.conv(bias_cd15, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{132},{0.16261771321296692},{-21.465538024902344},{20.001977920532227}}, "MobilenetV2/expanded_conv_15/project/add_fold#227");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (224);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{224}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007369794766418636},{-inf},{inf}}, "MobilenetV2/expanded_conv_15/project/add_fold#150_bias#152");
    auto bias_c31 = om.bias(conv31, biasWeights31, mv::DType("UInt8"), {{132},{0.16261771321296692},{-21.465538024902344},{20.001977920532227}});

    auto eltwise9 = om.eltwise({bias_c31,eltwise8}, "Add", mv::DType("UInt8"), {{134},{0.2111864686012268},{-28.298986434936523},{25.553564071655273}}, "MobilenetV2/expanded_conv_15/add#228");

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (1*1*224*1344);
    auto weights32 = om.constantInt(weightsData32,{1,1,224,1344}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.002720279386267066},{-0.33222347497940063},{0.35872751474380493}}, "MobilenetV2/expanded_conv_16/expand/Relu6#154_weights#155");
    auto conv32 = om.conv(eltwise9, weights32, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_16/expand/Relu6#229");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (1344);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005744862137362361},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/expand/Relu6#154_bias#156");
    auto bias_c32 = om.bias(conv32, biasWeights32, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData16 = mv::utils::generateSequence<int64_t> (3*3*1344*1);
    auto d_weights16 = om.constantInt(d_weightsData16,{3,3,1344,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{199},{0.17229272425174713},{-34.09807586669922},{9.664277076721191}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#157_weights#158");
    auto depthConv16 = om.depthwiseConv(bias_c32, d_weights16, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#230");

    std::vector<int64_t> biasd_WeightsData16 = mv::utils::generateSequence<int64_t> (1344);
    auto biasdWeights16 = om.constantInt(biasd_WeightsData16,{1344}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.004053785465657711},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/depthwise/Relu6#157_bias#159");
    auto bias_cd16 = om.bias(depthConv16, biasdWeights16, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*1344*448);
    auto weights33 = om.constantInt(weightsData33,{1,1,1344,448}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{148},{0.0067400760017335415},{-0.9913271069526672},{0.720652163028717}}, "MobilenetV2/expanded_conv_16/project/add_fold#160_weights#161");
    auto conv33 = om.conv(bias_cd16, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{126},{0.10853544622659683},{-13.675466537475586},{14.00107192993164}}, "MobilenetV2/expanded_conv_16/project/add_fold#231");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (448);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{448}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000158583716256544},{-inf},{inf}}, "MobilenetV2/expanded_conv_16/project/add_fold#160_bias#162");
    auto bias_c33 = om.bias(conv33, biasWeights33, mv::DType("UInt8"), {{126},{0.10853544622659683},{-13.675466537475586},{14.00107192993164}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*448*1792);
    auto weights34 = om.constantInt(weightsData34,{1,1,448,1792}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{111},{0.006747676059603691},{-0.7412212491035461},{0.9726884961128235}}, "MobilenetV2/Conv_1/Relu6#163_weights#164");
    auto conv34 = om.conv(bias_c33, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv_1/Relu6#232");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (1792);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{1792}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000732362037524581},{-inf},{inf}}, "MobilenetV2/Conv_1/Relu6#163_bias#165");
    auto bias_c34 = om.bias(conv34, biasWeights34, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    auto pool0 = om.averagePool(bias_c34, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Logits/AvgPool#233");

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (1*1*1792*1001);
    auto weights35 = om.constantInt(weightsData35,{1,1,1792,1001}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{104},{0.0017129590269178152},{-0.17631441354751587},{0.2587771713733673}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#167_weights#168");
    auto conv35 = om.conv(pool0, weights35, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{83},{0.1308872401714325},{-10.863640785217285},{22.512603759765625}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#234");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (1001);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{1001}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{4.030331547255628e-05},{-inf},{inf}}, "MobilenetV2/Logits/Conv2d_1c_1x1/act_quant/FakeQuantWithMinMaxVars#167_bias#169");
    auto bias_c35 = om.bias(conv35, biasWeights35, mv::DType("UInt8"), {{83},{0.1308872401714325},{-10.863640785217285},{22.512603759765625}});

    om.output(bias_c35);

    unit.initialize();
    unit.run();
}