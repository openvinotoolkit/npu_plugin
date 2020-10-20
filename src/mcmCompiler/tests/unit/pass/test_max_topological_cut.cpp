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

    auto input = om.input("", {112, 224, 3, 1}, mv::DType("UInt8"), mv::Order("NCHW"));
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(7*7*3*64);
    auto weights = om.constantInt("", weightsData, {7, 7, 3, 64}, mv::DType("UInt8"), mv::Order("NCWH"));
    auto conv = om.conv("", input, weights, {2, 2}, {3, 3, 3, 3});
    om.output("", conv);
    
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
    auto input0 = om.input("input#180", {224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{128},{0.007843137718737125},{-1.0},{1.0}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (7*7*3*64);
    auto weights0 = om.constantInt("conv1_weights#1", weightsData0,{7,7,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights0->setQuantParams({{115},{0.002871257718652487},{-0.32948583364486694},{0.40268489718437195}});
    auto conv0 = om.conv("conv1#181", input0, weights0, {2, 2}, {2, 3, 2, 3}, 1, 1);
    conv0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt("conv1_bias#2", biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights0->setQuantParams({{0},{2.2519669073517434e-05},{-inf},{inf}});
    auto bias_c0 = om.bias("", conv0, biasWeights0);
    bias_c0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool("pool1/max_pool#182", bias_c0, {3, 3}, {2, 2}, {0, 1, 0, 1}, true);
    pool0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights1 = om.constantInt("res2a_branch1_weights#5", weightsData1,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights1->setQuantParams({{137},{0.0030952668748795986},{-0.42331647872924805},{0.36597657203674316}});
    auto conv1 = om.conv("res2a_branch1#183", pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights1 = om.constantInt("res2a_branch1_bias#6", biasWeightsData1,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights1->setQuantParams({{0},{1.2138301826780662e-05},{-inf},{inf}});
    auto bias_c1 = om.bias("", conv1, biasWeights1);
    bias_c1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt("res2a_branch2a_weights#8", weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights2->setQuantParams({{138},{0.0029387609101831913},{-0.404168039560318},{0.34521597623825073}});
    auto conv2 = om.conv("res2a_branch2a#184", pool0, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt("res2a_branch2a_bias#9", biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights2->setQuantParams({{0},{1.1524552064656746e-05},{-inf},{inf}});
    auto bias_c2 = om.bias("", conv2, biasWeights2);
    bias_c2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights3 = om.constantInt("res2a_branch2b_weights#11", weightsData3,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights3->setQuantParams({{133},{0.0032645531464368105},{-0.43268874287605286},{0.3997723162174225}});
    auto conv3 = om.conv("res2a_branch2b#185", bias_c2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv3->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights3 = om.constantInt("res2a_branch2b_bias#12", biasWeightsData3,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights3->setQuantParams({{0},{1.2802169294445775e-05},{-inf},{inf}});
    auto bias_c3 = om.bias("", conv3, biasWeights3);
    bias_c3->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights4 = om.constantInt("res2a_branch2c_weights#14", weightsData4,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights4->setQuantParams({{130},{0.0032063836697489023},{-0.4181155264377594},{0.39951232075691223}});
    auto conv4 = om.conv("res2a_branch2c#186", bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv4->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt("res2a_branch2c_bias#15", biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights4->setQuantParams({{0},{1.2574053471325897e-05},{-inf},{inf}});
    auto bias_c4 = om.bias("", conv4, biasWeights4);
    bias_c4->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise0 = om.eltwise("", {bias_c1,bias_c4}, "Add");
    eltwise0->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights5 = om.constantInt("res2b_branch2a_weights#18", weightsData5,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights5->setQuantParams({{119},{0.0032081177923828363},{-0.3820513188838959},{0.436018705368042}});
    auto conv5 = om.conv("res2b_branch2a#188", eltwise0, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv5->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights5 = om.constantInt("res2b_branch2a_bias#19", biasWeightsData5,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights5->setQuantParams({{0},{1.2580853763211053e-05},{-inf},{inf}});
    auto bias_c5 = om.bias("", conv5, biasWeights5);
    bias_c5->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights6 = om.constantInt("res2b_branch2b_weights#21", weightsData6,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights6->setQuantParams({{123},{0.00342287658713758},{-0.41974323987960815},{0.45309028029441833}});
    auto conv6 = om.conv("res2b_branch2b#189", bias_c5, weights6, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv6->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights6 = om.constantInt("res2b_branch2b_bias#22", biasWeightsData6,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights6->setQuantParams({{0},{1.3423044947558083e-05},{-inf},{inf}});
    auto bias_c6 = om.bias("", conv6, biasWeights6);
    bias_c6->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights7 = om.constantInt("res2b_branch2c_weights#24", weightsData7,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights7->setQuantParams({{130},{0.0029558714013546705},{-0.38333263993263245},{0.3704145848751068}});
    auto conv7 = om.conv("res2b_branch2c#190", bias_c6, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv7->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights7 = om.constantInt("res2b_branch2c_bias#25", biasWeightsData7,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights7->setQuantParams({{0},{1.1591652764764149e-05},{-inf},{inf}});
    auto bias_c7 = om.bias("", conv7, biasWeights7);
    bias_c7->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise1 = om.eltwise("", {eltwise0,bias_c7}, "Add");
    eltwise1->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights8 = om.constantInt("res2c_branch2a_weights#28", weightsData8,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights8->setQuantParams({{127},{0.0031738088000565767},{-0.4021194577217102},{0.40720176696777344}});
    auto conv8 = om.conv("res2c_branch2a#192", eltwise1, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv8->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights8 = om.constantInt("res2c_branch2a_bias#29", biasWeightsData8,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights8->setQuantParams({{0},{1.2446308573998976e-05},{-inf},{inf}});
    auto bias_c8 = om.bias("", conv8, biasWeights8);
    bias_c8->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights9 = om.constantInt("res2c_branch2b_weights#31", weightsData9,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights9->setQuantParams({{132},{0.0030720513314008713},{-0.4059349000453949},{0.3774382174015045}});
    auto conv9 = om.conv("res2c_branch2b#193", bias_c8, weights9, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv9->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights9 = om.constantInt("res2c_branch2b_bias#32", biasWeightsData9,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights9->setQuantParams({{0},{1.204726049763849e-05},{-inf},{inf}});
    auto bias_c9 = om.bias("", conv9, biasWeights9);
    bias_c9->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights10 = om.constantInt("res2c_branch2c_weights#34", weightsData10,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights10->setQuantParams({{143},{0.0032446521800011396},{-0.46311089396476746},{0.36427542567253113}});
    auto conv10 = om.conv("res2c_branch2c#194", bias_c9, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv10->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights10 = om.constantInt("res2c_branch2c_bias#35", biasWeightsData10,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights10->setQuantParams({{0},{1.2724126463581342e-05},{-inf},{inf}});
    auto bias_c10 = om.bias("", conv10, biasWeights10);
    bias_c10->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise2 = om.eltwise("", {eltwise1,bias_c10}, "Add");
    eltwise2->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*256*512);
    auto weights11 = om.constantInt("res3a_branch1_weights#38", weightsData11,{1,1,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights11->setQuantParams({{122},{0.0034912910778075457},{-0.42566612362861633},{0.4646131098270416}});
    auto conv11 = om.conv("res3a_branch1#196", eltwise2, weights11, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv11->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights11 = om.constantInt("res3a_branch1_bias#39", biasWeightsData11,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights11->setQuantParams({{0},{1.369133769912878e-05},{-inf},{inf}});
    auto bias_c11 = om.bias("", conv11, biasWeights11);
    bias_c11->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*256*128);
    auto weights12 = om.constantInt("res3a_branch2a_weights#41", weightsData12,{1,1,256,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights12->setQuantParams({{135},{0.003216271521523595},{-0.43511319160461426},{0.38503605127334595}});
    auto conv12 = om.conv("res3a_branch2a#197", eltwise2, weights12, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv12->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights12 = om.constantInt("res3a_branch2a_bias#42", biasWeightsData12,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights12->setQuantParams({{0},{1.2612829777935985e-05},{-inf},{inf}});
    auto bias_c12 = om.bias("", conv12, biasWeights12);
    bias_c12->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights13 = om.constantInt("res3a_branch2b_weights#44", weightsData13,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights13->setQuantParams({{124},{0.003490086179226637},{-0.4312863051891327},{0.4586856961250305}});
    auto conv13 = om.conv("res3a_branch2b#198", bias_c12, weights13, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv13->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights13 = om.constantInt("res3a_branch2b_bias#45", biasWeightsData13,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights13->setQuantParams({{0},{1.368661287415307e-05},{-inf},{inf}});
    auto bias_c13 = om.bias("", conv13, biasWeights13);
    bias_c13->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights14 = om.constantInt("res3a_branch2c_weights#47", weightsData14,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights14->setQuantParams({{127},{0.003184879431501031},{-0.4049041271209717},{0.4072401225566864}});
    auto conv14 = om.conv("res3a_branch2c#199", bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv14->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights14 = om.constantInt("res3a_branch2c_bias#48", biasWeightsData14,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights14->setQuantParams({{0},{1.2489723303588107e-05},{-inf},{inf}});
    auto bias_c14 = om.bias("", conv14, biasWeights14);
    bias_c14->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise3 = om.eltwise("", {bias_c11,bias_c14}, "Add");
    eltwise3->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights15 = om.constantInt("res3b_branch2a_weights#51", weightsData15,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights15->setQuantParams({{129},{0.003236441407352686},{-0.4176654815673828},{0.4076271057128906}});
    auto conv15 = om.conv("res3b_branch2a#201", eltwise3, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv15->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights15 = om.constantInt("res3b_branch2a_bias#52", biasWeightsData15,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights15->setQuantParams({{0},{0.0001015354209812358},{-inf},{inf}});
    auto bias_c15 = om.bias("", conv15, biasWeights15);
    bias_c15->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights16 = om.constantInt("res3b_branch2b_weights#54", weightsData16,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights16->setQuantParams({{126},{0.003464744659140706},{-0.4370753765106201},{0.44643452763557434}});
    auto conv16 = om.conv("res3b_branch2b#202", bias_c15, weights16, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv16->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights16 = om.constantInt("res3b_branch2b_bias#55", biasWeightsData16,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights16->setQuantParams({{0},{1.3587234207079746e-05},{-inf},{inf}});
    auto bias_c16 = om.bias("", conv16, biasWeights16);
    bias_c16->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights17 = om.constantInt("res3b_branch2c_weights#57", weightsData17,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights17->setQuantParams({{123},{0.003219595178961754},{-0.3963310420513153},{0.4246657192707062}});
    auto conv17 = om.conv("res3b_branch2c#203", bias_c16, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv17->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights17 = om.constantInt("res3b_branch2c_bias#58", biasWeightsData17,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights17->setQuantParams({{0},{1.2625863746507093e-05},{-inf},{inf}});
    auto bias_c17 = om.bias("", conv17, biasWeights17);
    bias_c17->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise4 = om.eltwise("", {eltwise3,bias_c17}, "Add");
    eltwise4->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights18 = om.constantInt("res3c_branch2a_weights#61", weightsData18,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights18->setQuantParams({{135},{0.003667778568342328},{-0.4963448643684387},{0.4389386773109436}});
    auto conv18 = om.conv("res3c_branch2a#205", eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv18->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights18 = om.constantInt("res3c_branch2a_bias#62", biasWeightsData18,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights18->setQuantParams({{0},{0.00011506756709422916},{-inf},{inf}});
    auto bias_c18 = om.bias("", conv18, biasWeights18);
    bias_c18->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights19 = om.constantInt("res3c_branch2b_weights#64", weightsData19,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights19->setQuantParams({{129},{0.0035618969704955816},{-0.46099328994750977},{0.44729045033454895}});
    auto conv19 = om.conv("res3c_branch2b#206", bias_c18, weights19, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv19->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights19 = om.constantInt("res3c_branch2b_bias#65", biasWeightsData19,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights19->setQuantParams({{0},{1.3968223356641829e-05},{-inf},{inf}});
    auto bias_c19 = om.bias("", conv19, biasWeights19);
    bias_c19->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights20 = om.constantInt("res3c_branch2c_weights#67", weightsData20,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights20->setQuantParams({{129},{0.0033087413758039474},{-0.42709609866142273},{0.4166329503059387}});
    auto conv20 = om.conv("res3c_branch2c#207", bias_c19, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv20->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights20 = om.constantInt("res3c_branch2c_bias#68", biasWeightsData20,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights20->setQuantParams({{0},{1.2975456229469273e-05},{-inf},{inf}});
    auto bias_c20 = om.bias("", conv20, biasWeights20);
    bias_c20->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise5 = om.eltwise("", {eltwise4,bias_c20}, "Add");
    eltwise5->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights21 = om.constantInt("res3d_branch2a_weights#71", weightsData21,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights21->setQuantParams({{130},{0.0033619359601289034},{-0.43593573570251465},{0.4213579595088959}});
    auto conv21 = om.conv("res3d_branch2a#209", eltwise5, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv21->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights21 = om.constantInt("res3d_branch2a_bias#72", biasWeightsData21,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights21->setQuantParams({{0},{0.00010547250712988898},{-inf},{inf}});
    auto bias_c21 = om.bias("", conv21, biasWeights21);
    bias_c21->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights22 = om.constantInt("res3d_branch2b_weights#74", weightsData22,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights22->setQuantParams({{128},{0.003309462685137987},{-0.4247035086154938},{0.41920948028564453}});
    auto conv22 = om.conv("res3d_branch2b#210", bias_c21, weights22, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv22->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights22 = om.constantInt("res3d_branch2b_bias#75", biasWeightsData22,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights22->setQuantParams({{0},{1.2978284757991787e-05},{-inf},{inf}});
    auto bias_c22 = om.bias("", conv22, biasWeights22);
    bias_c22->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights23 = om.constantInt("res3d_branch2c_weights#77", weightsData23,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights23->setQuantParams({{134},{0.0036992442328482866},{-0.49393245577812195},{0.449374794960022}});
    auto conv23 = om.conv("res3d_branch2c#211", bias_c22, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv23->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights23 = om.constantInt("res3d_branch2c_bias#78", biasWeightsData23,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights23->setQuantParams({{0},{1.4506839761452284e-05},{-inf},{inf}});
    auto bias_c23 = om.bias("", conv23, biasWeights23);
    bias_c23->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise6 = om.eltwise("", {eltwise5,bias_c23}, "Add");
    eltwise6->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*512*1024);
    auto weights24 = om.constantInt("res4a_branch1_weights#81", weightsData24,{1,1,512,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights24->setQuantParams({{125},{0.0036697331815958023},{-0.459474116563797},{0.47630783915519714}});
    auto conv24 = om.conv("res4a_branch1#213", eltwise6, weights24, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv24->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights24 = om.constantInt("res4a_branch1_bias#82", biasWeightsData24,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights24->setQuantParams({{0},{0.00011512888158904389},{-inf},{inf}});
    auto bias_c24 = om.bias("", conv24, biasWeights24);
    bias_c24->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*512*256);
    auto weights25 = om.constantInt("res4a_branch2a_weights#84", weightsData25,{1,1,512,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights25->setQuantParams({{125},{0.0034257101360708475},{-0.42701420187950134},{0.4465419054031372}});
    auto conv25 = om.conv("res4a_branch2a#214", eltwise6, weights25, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv25->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights25 = om.constantInt("res4a_branch2a_bias#85", biasWeightsData25,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights25->setQuantParams({{0},{0.00010747326450655237},{-inf},{inf}});
    auto bias_c25 = om.bias("", conv25, biasWeights25);
    bias_c25->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights26 = om.constantInt("res4a_branch2b_weights#87", weightsData26,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights26->setQuantParams({{123},{0.0036150901578366756},{-0.4451659917831421},{0.47668200731277466}});
    auto conv26 = om.conv("res4a_branch2b#215", bias_c25, weights26, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv26->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights26 = om.constantInt("res4a_branch2b_bias#88", biasWeightsData26,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights26->setQuantParams({{0},{0.00011341459321556613},{-inf},{inf}});
    auto bias_c26 = om.bias("", conv26, biasWeights26);
    bias_c26->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights27 = om.constantInt("res4a_branch2c_weights#90", weightsData27,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights27->setQuantParams({{130},{0.0033847768791019917},{-0.44039711356163025},{0.4227209985256195}});
    auto conv27 = om.conv("res4a_branch2c#216", bias_c26, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv27->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights27 = om.constantInt("res4a_branch2c_bias#91", biasWeightsData27,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights27->setQuantParams({{0},{1.327363497694023e-05},{-inf},{inf}});
    auto bias_c27 = om.bias("", conv27, biasWeights27);
    bias_c27->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise7 = om.eltwise("", {bias_c24,bias_c27}, "Add");
    eltwise7->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights28 = om.constantInt("res4b_branch2a_weights#94", weightsData28,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights28->setQuantParams({{129},{0.004124246072024107},{-0.5335013270378113},{0.5181813836097717}});
    auto conv28 = om.conv("res4b_branch2a#218", eltwise7, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv28->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights28 = om.constantInt("res4b_branch2a_bias#95", biasWeightsData28,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights28->setQuantParams({{0},{0.00012938810687046498},{-inf},{inf}});
    auto bias_c28 = om.bias("", conv28, biasWeights28);
    bias_c28->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights29 = om.constantInt("res4b_branch2b_weights#97", weightsData29,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights29->setQuantParams({{129},{0.0037902493495494127},{-0.4906528890132904},{0.47586068511009216}});
    auto conv29 = om.conv("res4b_branch2b#219", bias_c28, weights29, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv29->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights29 = om.constantInt("res4b_branch2b_bias#98", biasWeightsData29,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights29->setQuantParams({{0},{0.000237819564063102},{-inf},{inf}});
    auto bias_c29 = om.bias("", conv29, biasWeights29);
    bias_c29->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights30 = om.constantInt("res4b_branch2c_weights#100", weightsData30,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights30->setQuantParams({{127},{0.003342666197568178},{-0.4256753921508789},{0.426704466342926}});
    auto conv30 = om.conv("res4b_branch2c#220", bias_c29, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv30->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights30 = om.constantInt("res4b_branch2c_bias#101", biasWeightsData30,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights30->setQuantParams({{0},{1.3108494385960512e-05},{-inf},{inf}});
    auto bias_c30 = om.bias("", conv30, biasWeights30);
    bias_c30->setQuantParams({{0},{0.0313725508749485},{0.0},{8.0}});

    auto eltwise8 = om.eltwise("", {eltwise7,bias_c30}, "Add");
    eltwise8->setQuantParams({{0},{0.0470588244497776},{0.0},{12.0}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights31 = om.constantInt("res4c_branch2a_weights#104", weightsData31,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights31->setQuantParams({{123},{0.003458196995779872},{-0.4249594211578369},{0.4568808376789093}});
    auto conv31 = om.conv("res4c_branch2a#222", eltwise8, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv31->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights31 = om.constantInt("res4c_branch2a_bias#105", biasWeightsData31,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights31->setQuantParams({{0},{0.0001627386809559539},{-inf},{inf}});
    auto bias_c31 = om.bias("", conv31, biasWeights31);
    bias_c31->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights32 = om.constantInt("res4c_branch2b_weights#107", weightsData32,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights32->setQuantParams({{130},{0.003711913013830781},{-0.48257341980934143},{0.46396440267562866}});
    auto conv32 = om.conv("res4c_branch2b#223", bias_c31, weights32, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv32->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights32 = om.constantInt("res4c_branch2b_bias#108", biasWeightsData32,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights32->setQuantParams({{0},{0.0002329043491045013},{-inf},{inf}});
    auto bias_c32 = om.bias("", conv32, biasWeights32);
    bias_c32->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights33 = om.constantInt("res4c_branch2c_weights#110", weightsData33,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights33->setQuantParams({{123},{0.003429228439927101},{-0.42193603515625},{0.45251724123954773}});
    auto conv33 = om.conv("res4c_branch2c#224", bias_c32, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv33->setQuantParams({{0},{0.0470588244497776},{0.0},{12.0}});

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights33 = om.constantInt("res4c_branch2c_bias#111", biasWeightsData33,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights33->setQuantParams({{0},{1.3447955097944941e-05},{-inf},{inf}});
    auto bias_c33 = om.bias("", conv33, biasWeights33);
    bias_c33->setQuantParams({{0},{0.0470588244497776},{0.0},{12.0}});

    auto eltwise9 = om.eltwise("", {eltwise8,bias_c33}, "Add");
    eltwise9->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights34 = om.constantInt("res4d_branch2a_weights#114", weightsData34,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights34->setQuantParams({{136},{0.0036149746738374233},{-0.49254778027534485},{0.42927077412605286}});
    auto conv34 = om.conv("res4d_branch2a#226", eltwise9, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv34->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights34 = om.constantInt("res4d_branch2a_bias#115", biasWeightsData34,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights34->setQuantParams({{0},{0.00022682193957734853},{-inf},{inf}});
    auto bias_c34 = om.bias("", conv34, biasWeights34);
    bias_c34->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights35 = om.constantInt("res4d_branch2b_weights#117", weightsData35,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights35->setQuantParams({{135},{0.00367855210788548},{-0.4958947002887726},{0.44213607907295227}});
    auto conv35 = om.conv("res4d_branch2b#227", bias_c34, weights35, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv35->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights35 = om.constantInt("res4d_branch2b_bias#118", biasWeightsData35,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights35->setQuantParams({{0},{0.00023081111430656165},{-inf},{inf}});
    auto bias_c35 = om.bias("", conv35, biasWeights35);
    bias_c35->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData36 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights36 = om.constantInt("res4d_branch2c_weights#120", weightsData36,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights36->setQuantParams({{134},{0.003840783378109336},{-0.5160320401191711},{0.4633677005767822}});
    auto conv36 = om.conv("res4d_branch2c#228", bias_c35, weights36, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv36->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    std::vector<int64_t> biasWeightsData36 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights36 = om.constantInt("res4d_branch2c_bias#121", biasWeightsData36,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights36->setQuantParams({{0},{1.5061895282997284e-05},{-inf},{inf}});
    auto bias_c36 = om.bias("", conv36, biasWeights36);
    bias_c36->setQuantParams({{0},{0.062745101749897},{0.0},{16.0}});

    auto eltwise10 = om.eltwise("", {eltwise9,bias_c36}, "Add");
    eltwise10->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData37 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights37 = om.constantInt("res4e_branch2a_weights#124", weightsData37,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights37->setQuantParams({{128},{0.0034344515297561884},{-0.43838825821876526},{0.4373968541622162}});
    auto conv37 = om.conv("res4e_branch2a#230", eltwise10, weights37, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv37->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> biasWeightsData37 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights37 = om.constantInt("res4e_branch2a_bias#125", biasWeightsData37,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights37->setQuantParams({{0},{0.0003232424787711352},{-inf},{inf}});
    auto bias_c37 = om.bias("", conv37, biasWeights37);
    bias_c37->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData38 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights38 = om.constantInt("res4e_branch2b_weights#127", weightsData38,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights38->setQuantParams({{125},{0.003688796190544963},{-0.4599181115627289},{0.48072493076324463}});
    auto conv38 = om.conv("res4e_branch2b#231", bias_c37, weights38, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv38->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData38 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights38 = om.constantInt("res4e_branch2b_bias#128", biasWeightsData38,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights38->setQuantParams({{0},{0.00034718081587925553},{-inf},{inf}});
    auto bias_c38 = om.bias("", conv38, biasWeights38);
    bias_c38->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData39 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights39 = om.constantInt("res4e_branch2c_weights#130", weightsData39,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights39->setQuantParams({{125},{0.0035333456471562386},{-0.44150012731552124},{0.4595029950141907}});
    auto conv39 = om.conv("res4e_branch2c#232", bias_c38, weights39, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv39->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> biasWeightsData39 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights39 = om.constantInt("res4e_branch2c_bias#131", biasWeightsData39,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights39->setQuantParams({{0},{1.3856257282895967e-05},{-inf},{inf}});
    auto bias_c39 = om.bias("", conv39, biasWeights39);
    bias_c39->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    auto eltwise11 = om.eltwise("", {eltwise10,bias_c39}, "Add");
    eltwise11->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData40 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights40 = om.constantInt("res4f_branch2a_weights#134", weightsData40,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights40->setQuantParams({{126},{0.0035140542313456535},{-0.4410395920276642},{0.4550442397594452}});
    auto conv40 = om.conv("res4f_branch2a#234", eltwise11, weights40, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv40->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> biasWeightsData40 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights40 = om.constantInt("res4f_branch2a_bias#135", biasWeightsData40,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights40->setQuantParams({{0},{0.00033073450322262943},{-inf},{inf}});
    auto bias_c40 = om.bias("", conv40, biasWeights40);
    bias_c40->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData41 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights41 = om.constantInt("res4f_branch2b_weights#137", weightsData41,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights41->setQuantParams({{123},{0.0035706025082618},{-0.4375752806663513},{0.47292837500572205}});
    auto conv41 = om.conv("res4f_branch2b#235", bias_c40, weights41, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv41->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData41 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights41 = om.constantInt("res4f_branch2b_bias#138", biasWeightsData41,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights41->setQuantParams({{0},{0.0003360567206982523},{-inf},{inf}});
    auto bias_c41 = om.bias("", conv41, biasWeights41);
    bias_c41->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData42 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights42 = om.constantInt("res4f_branch2c_weights#140", weightsData42,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights42->setQuantParams({{122},{0.0037916346918791533},{-0.46117961406707764},{0.5056872367858887}});
    auto conv42 = om.conv("res4f_branch2c#236", bias_c41, weights42, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv42->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> biasWeightsData42 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights42 = om.constantInt("res4f_branch2c_bias#141", biasWeightsData42,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights42->setQuantParams({{0},{1.4869156075292267e-05},{-inf},{inf}});
    auto bias_c42 = om.bias("", conv42, biasWeights42);
    bias_c42->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    auto eltwise12 = om.eltwise("", {eltwise11,bias_c42}, "Add");
    eltwise12->setQuantParams({{0},{0.0941176488995552},{0.0},{24.0}});

    std::vector<int64_t> weightsData43 = mv::utils::generateSequence<int64_t> (1*1*1024*2048);
    auto weights43 = om.constantInt("res5a_branch1_weights#144", weightsData43,{1,1,1024,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights43->setQuantParams({{128},{0.003836166113615036},{-0.4922131896018982},{0.4860091507434845}});
    auto conv43 = om.conv("res5a_branch1#238", eltwise12, weights43, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv43->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> biasWeightsData43 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights43 = om.constantInt("res5a_branch1_bias#145", biasWeightsData43,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights43->setQuantParams({{0},{0.0003610509156715125},{-inf},{inf}});
    auto bias_c43 = om.bias("", conv43, biasWeights43);
    bias_c43->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData44 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    auto weights44 = om.constantInt("res5a_branch2a_weights#147", weightsData44,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights44->setQuantParams({{121},{0.0036368558648973703},{-0.4405672550201416},{0.48683100938796997}});
    auto conv44 = om.conv("res5a_branch2a#239", eltwise12, weights44, {2, 2}, {0, 0, 0, 0}, 1, 1);
    conv44->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> biasWeightsData44 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights44 = om.constantInt("res5a_branch2a_bias#148", biasWeightsData44,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights44->setQuantParams({{0},{0.00034229233278892934},{-inf},{inf}});
    auto bias_c44 = om.bias("", conv44, biasWeights44);
    bias_c44->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData45 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights45 = om.constantInt("res5a_branch2b_weights#150", weightsData45,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights45->setQuantParams({{124},{0.004363630432635546},{-0.5415508151054382},{0.5711749792098999}});
    auto conv45 = om.conv("res5a_branch2b#240", bias_c44, weights45, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv45->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData45 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights45 = om.constantInt("res5a_branch2b_bias#151", biasWeightsData45,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights45->setQuantParams({{0},{0.0005475928774103522},{-inf},{inf}});
    auto bias_c45 = om.bias("", conv45, biasWeights45);
    bias_c45->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData46 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights46 = om.constantInt("res5a_branch2c_weights#153", weightsData46,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights46->setQuantParams({{126},{0.003843836486339569},{-0.4861159026622772},{0.4940624237060547}});
    auto conv46 = om.conv("res5a_branch2c#241", bias_c45, weights46, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv46->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> biasWeightsData46 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights46 = om.constantInt("res5a_branch2c_bias#154", biasWeightsData46,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights46->setQuantParams({{0},{1.5073868780746125e-05},{-inf},{inf}});
    auto bias_c46 = om.bias("", conv46, biasWeights46);
    bias_c46->setQuantParams({{0},{0.125490203499794},{0.0},{32.0}});

    auto eltwise13 = om.eltwise("", {bias_c43,bias_c46}, "Add");
    eltwise13->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData47 = mv::utils::generateSequence<int64_t> (1*1*2048*512);
    auto weights47 = om.constantInt("res5b_branch2a_weights#157", weightsData47,{1,1,2048,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights47->setQuantParams({{129},{0.0038684855680912733},{-0.497765451669693},{0.48869839310646057}});
    auto conv47 = om.conv("res5b_branch2a#243", eltwise13, weights47, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv47->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> biasWeightsData47 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights47 = om.constantInt("res5b_branch2a_bias#158", biasWeightsData47,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights47->setQuantParams({{0},{0.0006371623603627086},{-inf},{inf}});
    auto bias_c47 = om.bias("", conv47, biasWeights47);
    bias_c47->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData48 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights48 = om.constantInt("res5b_branch2b_weights#160", weightsData48,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights48->setQuantParams({{129},{0.004149848595261574},{-0.5362045764923096},{0.5220068097114563}});
    auto conv48 = om.conv("res5b_branch2b#244", bias_c47, weights48, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv48->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData48 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights48 = om.constantInt("res5b_branch2b_bias#161", biasWeightsData48,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights48->setQuantParams({{0},{0.0006835044478066266},{-inf},{inf}});
    auto bias_c48 = om.bias("", conv48, biasWeights48);
    bias_c48->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData49 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights49 = om.constantInt("res5b_branch2c_weights#163", weightsData49,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights49->setQuantParams({{143},{0.004270848352462053},{-0.6119490265846252},{0.4771173298358917}});
    auto conv49 = om.conv("res5b_branch2c#245", bias_c48, weights49, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv49->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> biasWeightsData49 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights49 = om.constantInt("res5b_branch2c_bias#164", biasWeightsData49,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights49->setQuantParams({{0},{1.674842496868223e-05},{-inf},{inf}});
    auto bias_c49 = om.bias("", conv49, biasWeights49);
    bias_c49->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    auto eltwise14 = om.eltwise("", {eltwise13,bias_c49}, "Add");
    eltwise14->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData50 = mv::utils::generateSequence<int64_t> (1*1*2048*512);
    auto weights50 = om.constantInt("res5c_branch2a_weights#167", weightsData50,{1,1,2048,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights50->setQuantParams({{129},{0.0037442538887262344},{-0.48447155952453613},{0.4703131914138794}});
    auto conv50 = om.conv("res5c_branch2a#247", eltwise14, weights50, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv50->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> biasWeightsData50 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights50 = om.constantInt("res5c_branch2a_bias#168", biasWeightsData50,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights50->setQuantParams({{0},{0.0006167006213217974},{-inf},{inf}});
    auto bias_c50 = om.bias("", conv50, biasWeights50);
    bias_c50->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> weightsData51 = mv::utils::generateSequence<int64_t> (3*3*512*512);
    auto weights51 = om.constantInt("res5c_branch2b_weights#170", weightsData51,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights51->setQuantParams({{127},{0.004263573791831732},{-0.5423005819320679},{0.544910728931427}});
    auto conv51 = om.conv("res5c_branch2b#248", bias_c50, weights51, {1, 1}, {1, 1, 1, 1}, 1, 1);
    conv51->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> biasWeightsData51 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights51 = om.constantInt("res5c_branch2b_bias#171", biasWeightsData51,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights51->setQuantParams({{0},{0.0007022356730885804},{-inf},{inf}});
    auto bias_c51 = om.bias("", conv51, biasWeights51);
    bias_c51->setQuantParams({{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData52 = mv::utils::generateSequence<int64_t> (1*1*512*2048);
    auto weights52 = om.constantInt("res5c_branch2c_weights#173", weightsData52,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights52->setQuantParams({{127},{0.0037484760396182537},{-0.47455498576164246},{0.48130640387535095}});
    auto conv52 = om.conv("res5c_branch2c#249", bias_c51, weights52, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv52->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    std::vector<int64_t> biasWeightsData52 = mv::utils::generateSequence<int64_t> (2048);
    auto biasWeights52 = om.constantInt("res5c_branch2c_bias#174", biasWeightsData52,{2048}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights52->setQuantParams({{0},{1.4699906387249939e-05},{-inf},{inf}});
    auto bias_c52 = om.bias("", conv52, biasWeights52);
    bias_c52->setQuantParams({{0},{0.16470588743686676},{0.0},{42.0}});

    auto eltwise15 = om.eltwise("", {eltwise14,bias_c52}, "Add");
    eltwise15->setQuantParams({{0},{0.250980406999588},{0.0},{64.0}});

    auto pool1 = om.averagePool("pool5/AvgPool#251", eltwise15, {7, 7}, {1, 1}, {0, 0, 0, 0}, true);
    pool1->setQuantParams({{0},{0.250980406999588},{0.0},{64.0}});

    std::vector<int64_t> weightsData53 = mv::utils::generateSequence<int64_t> (1*1*2048*1024);
    auto weights53 = om.constantInt("fc1000/fc1000_weights#178", weightsData53,{1,1,2048,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    weights53->setQuantParams({{130},{0.003821961348876357},{-0.49569910764694214},{0.4789010286331177}});
    auto conv53 = om.conv("fc1000/fc1000#252", pool1, weights53, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv53->setQuantParams({{0},{0.3294117748737335},{0.0},{84.0}});

    std::vector<int64_t> biasWeightsData53 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights53 = om.constantInt("fc1000/fc1000_bias#179", biasWeightsData53,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1));
    biasWeights53->setQuantParams({{0},{0.0009592373389750719},{-inf},{inf}});
    auto bias_c53 = om.bias("", conv53, biasWeights53);
    bias_c53->setQuantParams({{0},{0.3294117748737335},{0.0},{84.0}});

    om.output("", bias_c53);

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