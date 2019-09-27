#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

#include "include/mcm/base/json/json.hpp"

/*This test calculates max topological cut and does not perform partial serialisation as it is not required*/
TEST(MaxTopologicalCut, lessThanCMXMemory)
{
    mv::CompilationUnit unit("testMaxTopologicalCut");
    mv::OpModel& om = unit.model();

    auto input = om.input({112, 224, 3, 1}, mv::DType("UInt8"), mv::Order("NCHW"));
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(7*7*3*64);
    auto weights = om.constantInt(weightsData, {7, 7, 3, 64}, mv::DType("UInt8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {3, 3, 3, 3});
    om.output(conv);
    
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","GenerateWorkloads");
    unit.compilationDescriptor().remove("serialize","GenerateBlobKmb");
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "MemoryHack", false);

    unit.initialize();
    auto compOutput = unit.run();

    //mv::ControlModel cm(om);
    // auto output = cm.getOutput();
    // uint64_t maxTopologicalCutValue = 0;
    /*Get the max topological cut value*/
    // if(output->hasAttr("MaxTopologicalCutValue"))
    //     maxTopologicalCutValue = output->get<uint64_t>("MaxTopologicalCutValue");
    // else
    //     FAIL() << "MaxTopologicalCutValue missing from OpModel!";

    mv::DataModel dm(om);
    auto outflow = dm.getOutputFlow();
    uint64_t maxTopologicalCutValue = outflow->get<uint64_t>("MaxTopologicalCutValue");

    //using CompOutput
    //long long maxTopologicalCutValue = compOutput["passes"]["MaxTopologicalCutAndPartialSerialisation"]["MaxTopologicalCut"].get<long long>();
    //long long maxTopologicalCutValue = compOutput["maxTopologicalCut"].get<long long>();

    /*The max topological cut of the equivalent network in the PoC compiler is 492032*/
    ASSERT_EQ(maxTopologicalCutValue, 492032ll)  << "Fail: incorrect max cut value (" << maxTopologicalCutValue << ") compared to POC compiler";
}


TEST(MaxTopologicalCut, greaterThanCMXMemory)
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#180");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (7*7*3*64);
    auto weights0 = om.constantInt(weightsData0,{7,7,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.002871257718652487},{-0.32948583364486694},{0.40268489718437195}}, "conv1_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {2, 3, 2, 3}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#181");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2519669073517434e-05},{-inf},{inf}}, "conv1_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool(bias_c0, {3, 3}, {2, 2}, {0, 1, 0, 1}, true, "", "floor", {{0},{0.003921568859368563},{0.0},{1.0}}, "pool1/max_pool#182");

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights1 = om.constantInt(weightsData1,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.0030952668748795986},{-0.42331647872924805},{0.36597657203674316}}, "res2a_branch1_weights#5");
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch1#183");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2138301826780662e-05},{-inf},{inf}}, "res2a_branch1_bias#6");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{138},{0.0029387609101831913},{-0.404168039560318},{0.34521597623825073}}, "res2a_branch2a_weights#8");
    auto conv2 = om.conv(pool0, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2a#184");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1524552064656746e-05},{-inf},{inf}}, "res2a_branch2a_bias#9");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights3 = om.constantInt(weightsData3,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.0032645531464368105},{-0.43268874287605286},{0.3997723162174225}}, "res2a_branch2b_weights#11");
    auto conv3 = om.conv(bias_c2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2b#185");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2802169294445775e-05},{-inf},{inf}}, "res2a_branch2b_bias#12");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights4 = om.constantInt(weightsData4,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0032063836697489023},{-0.4181155264377594},{0.39951232075691223}}, "res2a_branch2c_weights#14");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2c#186");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2574053471325897e-05},{-inf},{inf}}, "res2a_branch2c_bias#15");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise0 = om.add({bias_c1,bias_c4}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a/Relu#187");

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights5 = om.constantInt(weightsData5,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.0032081177923828363},{-0.3820513188838959},{0.436018705368042}}, "res2b_branch2a_weights#18");
    auto conv5 = om.conv(eltwise0, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2a#188");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2580853763211053e-05},{-inf},{inf}}, "res2b_branch2a_bias#19");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights6 = om.constantInt(weightsData6,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.00342287658713758},{-0.41974323987960815},{0.45309028029441833}}, "res2b_branch2b_weights#21");
    auto conv6 = om.conv(bias_c5, weights6, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2b#189");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3423044947558083e-05},{-inf},{inf}}, "res2b_branch2b_bias#22");
    auto bias_c6 = om.bias(conv6, biasWeights6, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights7 = om.constantInt(weightsData7,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0029558714013546705},{-0.38333263993263245},{0.3704145848751068}}, "res2b_branch2c_weights#24");
    auto conv7 = om.conv(bias_c6, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2c#190");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1591652764764149e-05},{-inf},{inf}}, "res2b_branch2c_bias#25");
    auto bias_c7 = om.bias(conv7, biasWeights7, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise1 = om.add({eltwise0,bias_c7}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b/Relu#191");

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights8 = om.constantInt(weightsData8,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0031738088000565767},{-0.4021194577217102},{0.40720176696777344}}, "res2c_branch2a_weights#28");
    auto conv8 = om.conv(eltwise1, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2a#192");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2446308573998976e-05},{-inf},{inf}}, "res2c_branch2a_bias#29");
    auto bias_c8 = om.bias(conv8, biasWeights8, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights9 = om.constantInt(weightsData9,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.0030720513314008713},{-0.4059349000453949},{0.3774382174015045}}, "res2c_branch2b_weights#31");
    auto conv9 = om.conv(bias_c8, weights9, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2b#193");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.204726049763849e-05},{-inf},{inf}}, "res2c_branch2b_bias#32");
    auto bias_c9 = om.bias(conv9, biasWeights9, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights10 = om.constantInt(weightsData10,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.0032446521800011396},{-0.46311089396476746},{0.36427542567253113}}, "res2c_branch2c_weights#34");
    auto conv10 = om.conv(bias_c9, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2c#194");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2724126463581342e-05},{-inf},{inf}}, "res2c_branch2c_bias#35");
    auto bias_c10 = om.bias(conv10, biasWeights10, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise2 = om.add({eltwise1,bias_c10}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c/Relu#195");

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*256*512);
    auto weights11 = om.constantInt(weightsData11,{1,1,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0034912910778075457},{-0.42566612362861633},{0.4646131098270416}}, "res3a_branch1_weights#38");
    auto conv11 = om.conv(eltwise2, weights11, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch1#196");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.369133769912878e-05},{-inf},{inf}}, "res3a_branch1_bias#39");
    auto bias_c11 = om.bias(conv11, biasWeights11, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*256*128);
    auto weights12 = om.constantInt(weightsData12,{1,1,256,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003216271521523595},{-0.43511319160461426},{0.38503605127334595}}, "res3a_branch2a_weights#41");
    auto conv12 = om.conv(eltwise2, weights12, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2a#197");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2612829777935985e-05},{-inf},{inf}}, "res3a_branch2a_bias#42");
    auto bias_c12 = om.bias(conv12, biasWeights12, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights13 = om.constantInt(weightsData13,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003490086179226637},{-0.4312863051891327},{0.4586856961250305}}, "res3a_branch2b_weights#44");
    auto conv13 = om.conv(bias_c12, weights13, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2b#198");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.368661287415307e-05},{-inf},{inf}}, "res3a_branch2b_bias#45");
    auto bias_c13 = om.bias(conv13, biasWeights13, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights14 = om.constantInt(weightsData14,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003184879431501031},{-0.4049041271209717},{0.4072401225566864}}, "res3a_branch2c_weights#47");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2c#199");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2489723303588107e-05},{-inf},{inf}}, "res3a_branch2c_bias#48");
    auto bias_c14 = om.bias(conv14, biasWeights14, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise3 = om.add({bias_c11,bias_c14}, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3a/Relu#200");

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights15 = om.constantInt(weightsData15,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.003236441407352686},{-0.4176654815673828},{0.4076271057128906}}, "res3b_branch2a_weights#51");
    auto conv15 = om.conv(eltwise3, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b_branch2a#201");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001015354209812358},{-inf},{inf}}, "res3b_branch2a_bias#52");
    auto bias_c15 = om.bias(conv15, biasWeights15, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights16 = om.constantInt(weightsData16,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.003464744659140706},{-0.4370753765106201},{0.44643452763557434}}, "res3b_branch2b_weights#54");
    auto conv16 = om.conv(bias_c15, weights16, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b_branch2b#202");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3587234207079746e-05},{-inf},{inf}}, "res3b_branch2b_bias#55");
    auto bias_c16 = om.bias(conv16, biasWeights16, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights17 = om.constantInt(weightsData17,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.003219595178961754},{-0.3963310420513153},{0.4246657192707062}}, "res3b_branch2c_weights#57");
    auto conv17 = om.conv(bias_c16, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3b_branch2c#203");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2625863746507093e-05},{-inf},{inf}}, "res3b_branch2c_bias#58");
    auto bias_c17 = om.bias(conv17, biasWeights17, {{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise4 = om.add({eltwise3,bias_c17}, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3b/Relu#204");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights18 = om.constantInt(weightsData18,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003667778568342328},{-0.4963448643684387},{0.4389386773109436}}, "res3c_branch2a_weights#61");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c_branch2a#205");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00011506756709422916},{-inf},{inf}}, "res3c_branch2a_bias#62");
    auto bias_c18 = om.bias(conv18, biasWeights18, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights19 = om.constantInt(weightsData19,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0035618969704955816},{-0.46099328994750977},{0.44729045033454895}}, "res3c_branch2b_weights#64");
    auto conv19 = om.conv(bias_c18, weights19, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c_branch2b#206");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3968223356641829e-05},{-inf},{inf}}, "res3c_branch2b_bias#65");
    auto bias_c19 = om.bias(conv19, biasWeights19, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights20 = om.constantInt(weightsData20,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0033087413758039474},{-0.42709609866142273},{0.4166329503059387}}, "res3c_branch2c_weights#67");
    auto conv20 = om.conv(bias_c19, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3c_branch2c#207");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2975456229469273e-05},{-inf},{inf}}, "res3c_branch2c_bias#68");
    auto bias_c20 = om.bias(conv20, biasWeights20, {{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise5 = om.add({eltwise4,bias_c20}, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3c/Relu#208");

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights21 = om.constantInt(weightsData21,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0033619359601289034},{-0.43593573570251465},{0.4213579595088959}}, "res3d_branch2a_weights#71");
    auto conv21 = om.conv(eltwise5, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d_branch2a#209");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00010547250712988898},{-inf},{inf}}, "res3d_branch2a_bias#72");
    auto bias_c21 = om.bias(conv21, biasWeights21, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights22 = om.constantInt(weightsData22,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.003309462685137987},{-0.4247035086154938},{0.41920948028564453}}, "res3d_branch2b_weights#74");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d_branch2b#210");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2978284757991787e-05},{-inf},{inf}}, "res3d_branch2b_bias#75");
    auto bias_c22 = om.bias(conv22, biasWeights22, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights23 = om.constantInt(weightsData23,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0036992442328482866},{-0.49393245577812195},{0.449374794960022}}, "res3d_branch2c_weights#77");
    auto conv23 = om.conv(bias_c22, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3d_branch2c#211");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4506839761452284e-05},{-inf},{inf}}, "res3d_branch2c_bias#78");
    auto bias_c23 = om.bias(conv23, biasWeights23, {{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise6 = om.add({eltwise5,bias_c23}, {{0},{0.0313725508749485},{0.0},{8.0}}, "res3d/Relu#212");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*512*1024);
    auto weights24 = om.constantInt(weightsData24,{1,1,512,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0036697331815958023},{-0.459474116563797},{0.47630783915519714}}, "res4a_branch1_weights#81");
    auto conv24 = om.conv(eltwise6, weights24, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res4a_branch1#213");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00011512888158904389},{-inf},{inf}}, "res4a_branch1_bias#82");
    auto bias_c24 = om.bias(conv24, biasWeights24, {{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*512*256);
    auto weights25 = om.constantInt(weightsData25,{1,1,512,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0034257101360708475},{-0.42701420187950134},{0.4465419054031372}}, "res4a_branch2a_weights#84");
    auto conv25 = om.conv(eltwise6, weights25, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res4a_branch2a#214");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00010747326450655237},{-inf},{inf}}, "res4a_branch2a_bias#85");
    auto bias_c25 = om.bias(conv25, biasWeights25, {{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights26 = om.constantInt(weightsData26,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0036150901578366756},{-0.4451659917831421},{0.47668200731277466}}, "res4a_branch2b_weights#87");
    auto conv26 = om.conv(bias_c25, weights26, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a_branch2b#215");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00011341459321556613},{-inf},{inf}}, "res4a_branch2b_bias#88");
    auto bias_c26 = om.bias(conv26, biasWeights26, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights27 = om.constantInt(weightsData27,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0033847768791019917},{-0.44039711356163025},{0.4227209985256195}}, "res4a_branch2c_weights#90");
    auto conv27 = om.conv(bias_c26, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res4a_branch2c#216");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.327363497694023e-05},{-inf},{inf}}, "res4a_branch2c_bias#91");
    auto bias_c27 = om.bias(conv27, biasWeights27, {{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise7 = om.add({bias_c24,bias_c27}, {{0},{0.0313725508749485},{0.0},{8.0}}, "res4a/Relu#217");

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights28 = om.constantInt(weightsData28,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.004124246072024107},{-0.5335013270378113},{0.5181813836097717}}, "res4b_branch2a_weights#94");
    auto conv28 = om.conv(eltwise7, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.062745101749897},{0.0},{16.0}}, "res4b_branch2a#218");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00012938810687046498},{-inf},{inf}}, "res4b_branch2a_bias#95");
    auto bias_c28 = om.bias(conv28, biasWeights28, {{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights29 = om.constantInt(weightsData29,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0037902493495494127},{-0.4906528890132904},{0.47586068511009216}}, "res4b_branch2b_weights#97");
    auto conv29 = om.conv(bias_c28, weights29, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4b_branch2b#219");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000237819564063102},{-inf},{inf}}, "res4b_branch2b_bias#98");
    auto bias_c29 = om.bias(conv29, biasWeights29, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights30 = om.constantInt(weightsData30,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003342666197568178},{-0.4256753921508789},{0.426704466342926}}, "res4b_branch2c_weights#100");
    auto conv30 = om.conv(bias_c29, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0313725508749485},{0.0},{8.0}}, "res4b_branch2c#220");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3108494385960512e-05},{-inf},{inf}}, "res4b_branch2c_bias#101");
    auto bias_c30 = om.bias(conv30, biasWeights30, {{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise8 = om.add({eltwise7,bias_c30}, {{0},{0.0470588244497776},{0.0},{12.0}}, "res4b/Relu#221");

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights31 = om.constantInt(weightsData31,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.003458196995779872},{-0.4249594211578369},{0.4568808376789093}}, "res4c_branch2a_weights#104");
    auto conv31 = om.conv(eltwise8, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.062745101749897},{0.0},{16.0}}, "res4c_branch2a#222");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001627386809559539},{-inf},{inf}}, "res4c_branch2a_bias#105");
    auto bias_c31 = om.bias(conv31, biasWeights31, {{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights32 = om.constantInt(weightsData32,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.003711913013830781},{-0.48257341980934143},{0.46396440267562866}}, "res4c_branch2b_weights#107");
    auto conv32 = om.conv(bias_c31, weights32, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4c_branch2b#223");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002329043491045013},{-inf},{inf}}, "res4c_branch2b_bias#108");
    auto bias_c32 = om.bias(conv32, biasWeights32, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights33 = om.constantInt(weightsData33,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.003429228439927101},{-0.42193603515625},{0.45251724123954773}}, "res4c_branch2c_weights#110");
    auto conv33 = om.conv(bias_c32, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0470588244497776},{0.0},{12.0}}, "res4c_branch2c#224");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3447955097944941e-05},{-inf},{inf}}, "res4c_branch2c_bias#111");
    auto bias_c33 = om.bias(conv33, biasWeights33, {{0},{0.0470588244497776},{0.0},{12.0}});

    auto eltwise9 = om.add({eltwise8,bias_c33}, {{0},{0.062745101749897},{0.0},{16.0}}, "res4c/Relu#225");

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights34 = om.constantInt(weightsData34,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{136},{0.0036149746738374233},{-0.49254778027534485},{0.42927077412605286}}, "res4d_branch2a_weights#114");
    auto conv34 = om.conv(eltwise9, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.062745101749897},{0.0},{16.0}}, "res4d_branch2a#226");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022682193957734853},{-inf},{inf}}, "res4d_branch2a_bias#115");
    auto bias_c34 = om.bias(conv34, biasWeights34, {{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights35 = om.constantInt(weightsData35,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.00367855210788548},{-0.4958947002887726},{0.44213607907295227}}, "res4d_branch2b_weights#117");
    auto conv35 = om.conv(bias_c34, weights35, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4d_branch2b#227");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00023081111430656165},{-inf},{inf}}, "res4d_branch2b_bias#118");
    auto bias_c35 = om.bias(conv35, biasWeights35, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData36 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights36 = om.constantInt(weightsData36,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.003840783378109336},{-0.5160320401191711},{0.4633677005767822}}, "res4d_branch2c_weights#120");
    auto conv36 = om.conv(bias_c35, weights36, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.062745101749897},{0.0},{16.0}}, "res4d_branch2c#228");

    std::vector<int64_t> biasWeightsData36 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights36 = om.constantInt(biasWeightsData36,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5061895282997284e-05},{-inf},{inf}}, "res4d_branch2c_bias#121");
    auto bias_c36 = om.bias(conv36, biasWeights36, {{0},{0.062745101749897},{0.0},{16.0}});

    auto eltwise10 = om.add({eltwise9,bias_c36}, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4d/Relu#229");

    std::vector<int64_t> weightsData37 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights37 = om.constantInt(weightsData37,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0034344515297561884},{-0.43838825821876526},{0.4373968541622162}}, "res4e_branch2a_weights#124");
    auto conv37 = om.conv(eltwise10, weights37, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4e_branch2a#230");

    std::vector<int64_t> biasWeightsData37 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights37 = om.constantInt(biasWeightsData37,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003232424787711352},{-inf},{inf}}, "res4e_branch2a_bias#125");
    auto bias_c37 = om.bias(conv37, biasWeights37, {{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData38 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights38 = om.constantInt(weightsData38,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.003688796190544963},{-0.4599181115627289},{0.48072493076324463}}, "res4e_branch2b_weights#127");
    auto conv38 = om.conv(bias_c37, weights38, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4e_branch2b#231");

    std::vector<int64_t> biasWeightsData38 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights38 = om.constantInt(biasWeightsData38,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00034718081587925553},{-inf},{inf}}, "res4e_branch2b_bias#128");
    auto bias_c38 = om.bias(conv38, biasWeights38, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData39 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights39 = om.constantInt(weightsData39,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.0035333456471562386},{-0.44150012731552124},{0.4595029950141907}}, "res4e_branch2c_weights#130");
    auto conv39 = om.conv(bias_c38, weights39, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4e_branch2c#232");

    std::vector<int64_t> biasWeightsData39 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights39 = om.constantInt(biasWeightsData39,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3856257282895967e-05},{-inf},{inf}}, "res4e_branch2c_bias#131");
    auto bias_c39 = om.bias(conv39, biasWeights39, {{0},{0.0941176488995552},{0.0},{24.0}});

    auto eltwise11 = om.add({eltwise10,bias_c39}, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4e/Relu#233");

    std::vector<int64_t> weightsData40 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights40 = om.constantInt(weightsData40,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0035140542313456535},{-0.4410395920276642},{0.4550442397594452}}, "res4f_branch2a_weights#134");
    auto conv40 = om.conv(eltwise11, weights40, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4f_branch2a#234");

    std::vector<int64_t> biasWeightsData40 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights40 = om.constantInt(biasWeightsData40,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00033073450322262943},{-inf},{inf}}, "res4f_branch2a_bias#135");
    auto bias_c40 = om.bias(conv40, biasWeights40, {{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData41 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights41 = om.constantInt(weightsData41,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0035706025082618},{-0.4375752806663513},{0.47292837500572205}}, "res4f_branch2b_weights#137");
    auto conv41 = om.conv(bias_c40, weights41, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4f_branch2b#235");

    std::vector<int64_t> biasWeightsData41 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights41 = om.constantInt(biasWeightsData41,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003360567206982523},{-inf},{inf}}, "res4f_branch2b_bias#138");
    auto bias_c41 = om.bias(conv41, biasWeights41, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData42 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights42 = om.constantInt(weightsData42,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0037916346918791533},{-0.46117961406707764},{0.5056872367858887}}, "res4f_branch2c_weights#140");
    auto conv42 = om.conv(bias_c41, weights42, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4f_branch2c#236");

    std::vector<int64_t> biasWeightsData42 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights42 = om.constantInt(biasWeightsData42,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4869156075292267e-05},{-inf},{inf}}, "res4f_branch2c_bias#141");
    auto bias_c42 = om.bias(conv42, biasWeights42, {{0},{0.0941176488995552},{0.0},{24.0}});

    auto eltwise12 = om.add({eltwise11,bias_c42}, {{0},{0.0941176488995552},{0.0},{24.0}}, "res4f/Relu#237");

    std::vector<int64_t> weightsData43 = mv::utils::generateSequence<int64_t> (1*1*1024*2048);
    auto weights43 = om.constantInt(weightsData43,{1,1,1024,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.003836166113615036},{-0.4922131896018982},{0.4860091507434845}}, "res5a_branch1_weights#144");
    auto conv43 = om.conv(eltwise12, weights43, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "res5a_branch1#238");

    std::vector<int64_t> biasWeightsData43 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights43 = om.constantInt(biasWeightsData43,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003610509156715125},{-inf},{inf}}, "res5a_branch1_bias#145");
    auto bias_c43 = om.bias(conv43, biasWeights43, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData44 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    auto weights44 = om.constantInt(weightsData44,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.0036368558648973703},{-0.4405672550201416},{0.48683100938796997}}, "res5a_branch2a_weights#147");
    auto conv44 = om.conv(eltwise12, weights44, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "res5a_branch2a#239");

    std::vector<int64_t> biasWeightsData44 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights44 = om.constantInt(biasWeightsData44,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00034229233278892934},{-inf},{inf}}, "res5a_branch2a_bias#148");
    auto bias_c44 = om.bias(conv44, biasWeights44, {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData45 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights45 = om.constantInt(weightsData45,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.004363630432635546},{-0.5415508151054382},{0.5711749792098999}}, "res5a_branch2b_weights#150");
    auto conv45 = om.conv(bias_c44, weights45, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a_branch2b#240");

    std::vector<int64_t> biasWeightsData45 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights45 = om.constantInt(biasWeightsData45,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005475928774103522},{-inf},{inf}}, "res5a_branch2b_bias#151");
    auto bias_c45 = om.bias(conv45, biasWeights45, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData46 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights46 = om.constantInt(weightsData46,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.003843836486339569},{-0.4861159026622772},{0.4940624237060547}}, "res5a_branch2c_weights#153");
    auto conv46 = om.conv(bias_c45, weights46, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "res5a_branch2c#241");

    std::vector<int64_t> biasWeightsData46 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights46 = om.constantInt(biasWeightsData46,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5073868780746125e-05},{-inf},{inf}}, "res5a_branch2c_bias#154");
    auto bias_c46 = om.bias(conv46, biasWeights46, {{0},{0.125490203499794},{0.0},{32.0}});

    auto eltwise13 = om.add({bias_c43,bias_c46}, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5a/Relu#242");

    std::vector<int64_t> weightsData47 = mv::utils::generateSequence<int64_t> (1*1*2048*512);
    auto weights47 = om.constantInt(weightsData47,{1,1,2048,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0038684855680912733},{-0.497765451669693},{0.48869839310646057}}, "res5b_branch2a_weights#157");
    auto conv47 = om.conv(eltwise13, weights47, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5b_branch2a#243");

    std::vector<int64_t> biasWeightsData47 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights47 = om.constantInt(biasWeightsData47,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006371623603627086},{-inf},{inf}}, "res5b_branch2a_bias#158");
    auto bias_c47 = om.bias(conv47, biasWeights47, {{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData48 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights48 = om.constantInt(weightsData48,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.004149848595261574},{-0.5362045764923096},{0.5220068097114563}}, "res5b_branch2b_weights#160");
    auto conv48 = om.conv(bias_c47, weights48, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5b_branch2b#244");

    std::vector<int64_t> biasWeightsData48 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights48 = om.constantInt(biasWeightsData48,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006835044478066266},{-inf},{inf}}, "res5b_branch2b_bias#161");
    auto bias_c48 = om.bias(conv48, biasWeights48, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData49 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights49 = om.constantInt(weightsData49,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.004270848352462053},{-0.6119490265846252},{0.4771173298358917}}, "res5b_branch2c_weights#163");
    auto conv49 = om.conv(bias_c48, weights49, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5b_branch2c#245");

    std::vector<int64_t> biasWeightsData49 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights49 = om.constantInt(biasWeightsData49,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.674842496868223e-05},{-inf},{inf}}, "res5b_branch2c_bias#164");
    auto bias_c49 = om.bias(conv49, biasWeights49, {{0},{0.16470588743686676},{0.0},{42.0}});

    auto eltwise14 = om.add({eltwise13,bias_c49}, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5b/Relu#246");

    std::vector<int64_t> weightsData50 = mv::utils::generateSequence<int64_t> (1*1*2048*512);
    auto weights50 = om.constantInt(weightsData50,{1,1,2048,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0037442538887262344},{-0.48447155952453613},{0.4703131914138794}}, "res5c_branch2a_weights#167");
    auto conv50 = om.conv(eltwise14, weights50, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5c_branch2a#247");

    std::vector<int64_t> biasWeightsData50 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights50 = om.constantInt(biasWeightsData50,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006167006213217974},{-inf},{inf}}, "res5c_branch2a_bias#168");
    auto bias_c50 = om.bias(conv50, biasWeights50, {{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData51 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights51 = om.constantInt(weightsData51,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.004263573791831732},{-0.5423005819320679},{0.544910728931427}}, "res5c_branch2b_weights#170");
    auto conv51 = om.conv(bias_c50, weights51, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5c_branch2b#248");

    std::vector<int64_t> biasWeightsData51 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights51 = om.constantInt(biasWeightsData51,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007022356730885804},{-inf},{inf}}, "res5c_branch2b_bias#171");
    auto bias_c51 = om.bias(conv51, biasWeights51, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData52 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights52 = om.constantInt(weightsData52,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0037484760396182537},{-0.47455498576164246},{0.48130640387535095}}, "res5c_branch2c_weights#173");
    auto conv52 = om.conv(bias_c51, weights52, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5c_branch2c#249");

    std::vector<int64_t> biasWeightsData52 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights52 = om.constantInt(biasWeightsData52,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4699906387249939e-05},{-inf},{inf}}, "res5c_branch2c_bias#174");
    auto bias_c52 = om.bias(conv52, biasWeights52, {{0},{0.16470588743686676},{0.0},{42.0}});

    auto eltwise15 = om.add({eltwise14,bias_c52}, {{0},{0.250980406999588},{0.0},{64.0}}, "res5c/Relu#250");

    auto pool1 = om.averagePool(eltwise15, {7, 7}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{0},{0.250980406999588},{0.0},{64.0}}, "pool5/AvgPool#251");

    std::vector<int64_t> weightsData53 = mv::utils::generateSequence<int64_t> (1*1*2048*1024);
    auto weights53 = om.constantInt(weightsData53,{1,1,2048,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.003821961348876357},{-0.49569910764694214},{0.4789010286331177}}, "fc1000/fc1000_weights#178");
    auto conv53 = om.conv(pool1, weights53, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.3294117748737335},{0.0},{84.0}}, "fc1000/fc1000#252");

    std::vector<int64_t> biasWeightsData53 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights53 = om.constantInt(biasWeightsData53,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0009592373389750719},{-inf},{inf}}, "fc1000/fc1000_bias#179");
    auto bias_c53 = om.bias(conv53, biasWeights53, {{0},{0.3294117748737335},{0.0},{84.0}});

    om.output(bias_c53);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490-resnet50-multiclustering.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","GenerateWorkloads");
    unit.compilationDescriptor().remove("serialize","GenerateBlobKmb");
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "MemoryHack", false);
    unit.compilationDescriptor().remove("finalize", "TensorGraphColoring");

    unit.initialize();
    unit.run();
   
    mv::DataModel dm(om);
    auto outflow = dm.getOutputFlow();
    uint64_t maxTopologicalCutValue = outflow->get<uint64_t>("MaxTopologicalCutValue");
    
    ASSERT_EQ(maxTopologicalCutValue, 1491456) << "Fail: incorrect max cut value (" << maxTopologicalCutValue << ") compared to POC compiler";
}
/* output of POC for this test "greaterThanCMXMemory"
Network require 1432.0 kB (175.92% of available CMX memory)
Network require 1244.0 kB (152.83% of available CMX memory)
Network require 1236.0 kB (151.84% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1192.0 kB (146.44% of available CMX memory)
Network require 1172.0 kB (143.98% of available CMX memory)
Network require 1116.0 kB (137.10% of available CMX memory)
Network require 1064.0 kB (130.71% of available CMX memory)
Network require 1046.0 kB (128.50% of available CMX memory)
Network require 1036.0 kB (127.27% of available CMX memory)
Network require 1036.0 kB (127.27% of available CMX memory)
Network require 1036.0 kB (127.27% of available CMX memory)
Network require 1036.0 kB (127.27% of available CMX memory)
Network require 1036.0 kB (127.27% of available CMX memory)
Network require 1032.0 kB (126.78% of available CMX memory)
Network require 972.0 kB (119.41% of available CMX memory)
Network require 972.0 kB (119.41% of available CMX memory)
Network require 972.0 kB (119.41% of available CMX memory)
Network require 972.0 kB (119.41% of available CMX memory)
Network require 968.0 kB (118.92% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 952.0 kB (116.95% of available CMX memory)
Network require 948.0 kB (116.46% of available CMX memory)
Network require 948.0 kB (116.46% of available CMX memory)
Network require 948.0 kB (116.46% of available CMX memory)
Network require 948.0 kB (116.46% of available CMX memory)
Network require 948.0 kB (116.46% of available CMX memory)
Network require 940.0 kB (115.48% of available CMX memory)
Network require 939.0 kB (115.36% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 936.0 kB (114.99% of available CMX memory)
Network require 932.0 kB (114.50% of available CMX memory)
Network require 932.0 kB (114.50% of available CMX memory)
Network require 932.0 kB (114.50% of available CMX memory)
Network require 932.0 kB (114.50% of available CMX memory)
Network require 932.0 kB (114.50% of available CMX memory)
Network require 924.0 kB (113.51% of available CMX memory)
Network require 918.0 kB (112.78% of available CMX memory)
Network require 918.0 kB (112.78% of available CMX memory)
Network require 914.0 kB (112.29% of available CMX memory)
Network require 907.0 kB (111.43% of available CMX memory)
Network require 905.0 kB (111.18% of available CMX memory)
Network require 868.0 kB (106.63% of available CMX memory)
Network require 856.0 kB (105.16% of available CMX memory)
Network require 823.0 kB (101.11% of available CMX memory)
Network require 823.0 kB (101.11% of available CMX memory)
Network require 822.0 kB (100.98% of available CMX memory)
Out Network can be executed using dynamic scheduling. Max peak memory is 804.0 kB
*/