#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

#include "include/mcm/base/json/json.hpp"

/*This test calculates max topological cut and does not perform partial serialisation as it is not required*/
TEST(MaxTopologicalCut, lessThanCMXMemory)
{
    mv::CompilationUnit unit("testMaxTopologicalCut");
    mv::OpModel& om = unit.model();

    auto input = om.input({112, 224, 3, 1}, mv::DType("Float8"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(7*7*3*64);
    auto weights = om.constant(weightsData, {7, 7, 3, 64}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {3, 3, 3, 3});
    om.output(conv);
    
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","GenerateWorkloads");
    unit.compilationDescriptor().remove("serialize","GenerateBlobKeembay");
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
    auto input0 = om.input({125,125,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#156");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (7*7*3*64);
    auto weights0 = om.constantInt(weightsData0,{7,7,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.003092964645475149},{-0.35145992040634155},{0.43724608421325684}}, "conv1_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {3, 3, 3, 3}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#157");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.4258546545752324e-05},{-inf},{inf}}, "conv1_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool(bias_c0, {3, 3}, {2, 2}, {1, 1, 1, 1}, true, "", "floor", {{0},{0.003921568859368563},{0.0},{1.0}}, "pool1/max_pool#158");

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights1 = om.constantInt(weightsData1,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.00302461301907897},{-0.3878987431526184},{0.3833775818347931}}, "res2a_branch1_weights#5");
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch1#159");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1861227903864346e-05},{-inf},{inf}}, "res2a_branch1_bias#6");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.002752005122601986},{-0.3767995536327362},{0.32496178150177}}, "res2a_branch2a_weights#8");
    auto conv2 = om.conv(pool0, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2a#160");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0792177818075288e-05},{-inf},{inf}}, "res2a_branch2a_bias#9");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights3 = om.constantInt(weightsData3,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{117},{0.003247082931920886},{-0.38003775477409363},{0.44796839356422424}}, "res2a_branch2b_weights#11");
    auto conv3 = om.conv(bias_c2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2b#161");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2733658877550624e-05},{-inf},{inf}}, "res2a_branch2b_bias#12");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights4 = om.constantInt(weightsData4,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0030896198004484177},{-0.39415132999420166},{0.3937017321586609}}, "res2a_branch2c_weights#14");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2c#162");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2116156540287193e-05},{-inf},{inf}}, "res2a_branch2c_bias#15");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise0 = om.add(bias_c1,bias_c4, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a/Relu#163");

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights5 = om.constantInt(weightsData5,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{140},{0.0034382864832878113},{-0.4814651608467102},{0.39529791474342346}}, "res2b_branch2a_weights#18");
    auto conv5 = om.conv(eltwise0, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2a#164");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3483476323017385e-05},{-inf},{inf}}, "res2b_branch2a_bias#19");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights6 = om.constantInt(weightsData6,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0030729372519999743},{-0.391032338142395},{0.39256665110588074}}, "res2b_branch2b_weights#21");
    auto conv6 = om.conv(bias_c5, weights6, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2b#165");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2050733857904561e-05},{-inf},{inf}}, "res2b_branch2b_bias#22");
    auto bias_c6 = om.bias(conv6, biasWeights6, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights7 = om.constantInt(weightsData7,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.0033045473974198103},{-0.36916449666023254},{0.47349509596824646}}, "res2b_branch2c_weights#24");
    auto conv7 = om.conv(bias_c6, weights7, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2c#166");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2959009836777113e-05},{-inf},{inf}}, "res2b_branch2c_bias#25");
    auto bias_c7 = om.bias(conv7, biasWeights7, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise1 = om.add(eltwise0,bias_c7, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b/Relu#167");

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights8 = om.constantInt(weightsData8,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0028829395305365324},{-0.36190512776374817},{0.3732444643974304}}, "res2c_branch2a_weights#28");
    auto conv8 = om.conv(eltwise1, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2a#168");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.130564578488702e-05},{-inf},{inf}}, "res2c_branch2a_bias#29");
    auto bias_c8 = om.bias(conv8, biasWeights8, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights9 = om.constantInt(weightsData9,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{118},{0.0031152183655649424},{-0.36800646781921387},{0.426374226808548}}, "res2c_branch2b_weights#31");
    auto conv9 = om.conv(bias_c8, weights9, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2b#169");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2216542927490082e-05},{-inf},{inf}}, "res2c_branch2b_bias#32");
    auto bias_c9 = om.bias(conv9, biasWeights9, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights10 = om.constantInt(weightsData10,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.00304236588999629},{-0.36257898807525635},{0.41322433948516846}}, "res2c_branch2c_weights#34");
    auto conv10 = om.conv(bias_c9, weights10, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c_branch2c#170");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1930846994800959e-05},{-inf},{inf}}, "res2c_branch2c_bias#35");
    auto bias_c10 = om.bias(conv10, biasWeights10, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise2 = om.add(eltwise1,bias_c10, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2c/Relu#171");

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*256*512);
    auto weights11 = om.constantInt(weightsData11,{1,1,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0034173696767538786},{-0.4379696846008301},{0.433459609746933}}, "res3a_branch1_weights#38");
    auto conv11 = om.conv(eltwise2, weights11, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch1#172");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3401449905359186e-05},{-inf},{inf}}, "res3a_branch1_bias#39");
    auto bias_c11 = om.bias(conv11, biasWeights11, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*256*128);
    auto weights12 = om.constantInt(weightsData12,{1,1,256,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0030818446539342403},{-0.4166364371776581},{0.3692339360713959}}, "res3a_branch2a_weights#41");
    auto conv12 = om.conv(eltwise2, weights12, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2a#173");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2085664820915554e-05},{-inf},{inf}}, "res3a_branch2a_bias#42");
    auto bias_c12 = om.bias(conv12, biasWeights12, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights13 = om.constantInt(weightsData13,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0033836159855127335},{-0.4353307783603668},{0.42749130725860596}}, "res3a_branch2b_weights#44");
    auto conv13 = om.conv(bias_c12, weights13, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2b#174");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3269082046463154e-05},{-inf},{inf}}, "res3a_branch2b_bias#45");
    auto bias_c13 = om.bias(conv13, biasWeights13, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights14 = om.constantInt(weightsData14,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.0036475984379649162},{-0.45274806022644043},{0.47738951444625854}}, "res3a_branch2c_weights#47");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a_branch2c#175");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4304307114798576e-05},{-inf},{inf}}, "res3a_branch2c_bias#48");
    auto bias_c14 = om.bias(conv14, biasWeights14, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise3 = om.add(bias_c11,bias_c14, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3a/Relu#176");

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights15 = om.constantInt(weightsData15,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0034529808908700943},{-0.43392059206962585},{0.44658955931663513}}, "res3b_branch2a_weights#51");
    auto conv15 = om.conv(eltwise3, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b_branch2a#177");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3541101907321718e-05},{-inf},{inf}}, "res3b_branch2a_bias#52");
    auto bias_c15 = om.bias(conv15, biasWeights15, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights16 = om.constantInt(weightsData16,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0034454006236046553},{-0.44169384241104126},{0.4368833005428314}}, "res3b_branch2b_weights#54");
    auto conv16 = om.conv(bias_c15, weights16, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b_branch2b#178");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.351137507299427e-05},{-inf},{inf}}, "res3b_branch2b_bias#55");
    auto bias_c16 = om.bias(conv16, biasWeights16, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights17 = om.constantInt(weightsData17,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.003634415101259947},{-0.4827468693256378},{0.44402897357940674}}, "res3b_branch2c_weights#57");
    auto conv17 = om.conv(bias_c16, weights17, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b_branch2c#179");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4252607797970995e-05},{-inf},{inf}}, "res3b_branch2c_bias#58");
    auto bias_c17 = om.bias(conv17, biasWeights17, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise4 = om.add(eltwise3,bias_c17, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3b/Relu#180");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights18 = om.constantInt(weightsData18,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.003368707839399576},{-0.4082348644733429},{0.4507856070995331}}, "res3c_branch2a_weights#61");
    auto conv18 = om.conv(eltwise4, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c_branch2a#181");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3210618817538489e-05},{-inf},{inf}}, "res3c_branch2a_bias#62");
    auto bias_c18 = om.bias(conv18, biasWeights18, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights19 = om.constantInt(weightsData19,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{139},{0.0035863379016518593},{-0.4985380172729492},{0.4159781336784363}}, "res3c_branch2b_weights#64");
    auto conv19 = om.conv(bias_c18, weights19, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c_branch2b#182");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4064069546293467e-05},{-inf},{inf}}, "res3c_branch2b_bias#65");
    auto bias_c19 = om.bias(conv19, biasWeights19, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights20 = om.constantInt(weightsData20,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.00327823543921113},{-0.41498062014579773},{0.4209694266319275}}, "res3c_branch2c_weights#67");
    auto conv20 = om.conv(bias_c19, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c_branch2c#183");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2855825843871571e-05},{-inf},{inf}}, "res3c_branch2c_bias#68");
    auto bias_c20 = om.bias(conv20, biasWeights20, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise5 = om.add(eltwise4,bias_c20, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3c/Relu#184");

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (1*1*512*128);
    auto weights21 = om.constantInt(weightsData21,{1,1,512,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0031496756710112095},{-0.40471169352531433},{0.39845559000968933}}, "res3d_branch2a_weights#71");
    auto conv21 = om.conv(eltwise5, weights21, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d_branch2a#185");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2351669283816591e-05},{-inf},{inf}}, "res3d_branch2a_bias#72");
    auto bias_c21 = om.bias(conv21, biasWeights21, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights22 = om.constantInt(weightsData22,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0034975758753716946},{-0.4685528576374054},{0.4233289957046509}}, "res3d_branch2b_weights#74");
    auto conv22 = om.conv(bias_c21, weights22, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d_branch2b#186");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3715984096052125e-05},{-inf},{inf}}, "res3d_branch2b_bias#75");
    auto bias_c22 = om.bias(conv22, biasWeights22, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (1*1*128*512);
    auto weights23 = om.constantInt(weightsData23,{1,1,128,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003473477903753519},{-0.43159955739974976},{0.45413732528686523}}, "res3d_branch2c_weights#77");
    auto conv23 = om.conv(bias_c22, weights23, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d_branch2c#187");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3621482139569707e-05},{-inf},{inf}}, "res3d_branch2c_bias#78");
    auto bias_c23 = om.bias(conv23, biasWeights23, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise6 = om.add(eltwise5,bias_c23, {{0},{0.003921568859368563},{0.0},{1.0}}, "res3d/Relu#188");

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (1*1*512*1024);
    auto weights24 = om.constantInt(weightsData24,{1,1,512,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.003668911289423704},{-0.4485097825527191},{0.48706260323524475}}, "res4a_branch1_weights#81");
    auto conv24 = om.conv(eltwise6, weights24, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a_branch1#189");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4387887858902104e-05},{-inf},{inf}}, "res4a_branch1_bias#82");
    auto bias_c24 = om.bias(conv24, biasWeights24, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*512*256);
    auto weights25 = om.constantInt(weightsData25,{1,1,512,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003668983234092593},{-0.46737179160118103},{0.46821895241737366}}, "res4a_branch2a_weights#84");
    auto conv25 = om.conv(eltwise6, weights25, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a_branch2a#190");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4388169802259654e-05},{-inf},{inf}}, "res4a_branch2a_bias#85");
    auto bias_c25 = om.bias(conv25, biasWeights25, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights26 = om.constantInt(weightsData26,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0036061236169189215},{-0.4730028808116913},{0.4465586245059967}}, "res4a_branch2b_weights#87");
    auto conv26 = om.conv(bias_c25, weights26, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a_branch2b#191");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4141661267785821e-05},{-inf},{inf}}, "res4a_branch2b_bias#88");
    auto bias_c26 = om.bias(conv26, biasWeights26, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights27 = om.constantInt(weightsData27,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.003686725627630949},{-0.47700202465057373},{0.463113009929657}}, "res4a_branch2c_weights#90");
    auto conv27 = om.conv(bias_c26, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a_branch2c#192");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4457747965934686e-05},{-inf},{inf}}, "res4a_branch2c_bias#91");
    auto bias_c27 = om.bias(conv27, biasWeights27, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise7 = om.add(bias_c24,bias_c27, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4a/Relu#193");

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights28 = om.constantInt(weightsData28,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{139},{0.003774145618081093},{-0.526213526725769},{0.436193585395813}}, "res4b_branch2a_weights#94");
    auto conv28 = om.conv(eltwise7, weights28, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4b_branch2a#194");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.480057107983157e-05},{-inf},{inf}}, "res4b_branch2a_bias#95");
    auto bias_c28 = om.bias(conv28, biasWeights28, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights29 = om.constantInt(weightsData29,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.0035478430800139904},{-0.44775399565696716},{0.4569459855556488}}, "res4b_branch2b_weights#97");
    auto conv29 = om.conv(bias_c28, weights29, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4b_branch2b#195");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3913109796703793e-05},{-inf},{inf}}, "res4b_branch2b_bias#98");
    auto bias_c29 = om.bias(conv29, biasWeights29, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights30 = om.constantInt(weightsData30,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0037139183841645718},{-0.5010111927986145},{0.44603800773620605}}, "res4b_branch2c_weights#100");
    auto conv30 = om.conv(bias_c29, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4b_branch2c#196");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4564386219717562e-05},{-inf},{inf}}, "res4b_branch2c_bias#101");
    auto bias_c30 = om.bias(conv30, biasWeights30, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise8 = om.add(eltwise7,bias_c30, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4b/Relu#197");

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights31 = om.constantInt(weightsData31,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.003759232582524419},{-0.48674672842025757},{0.47185757756233215}}, "res4c_branch2a_weights#104");
    auto conv31 = om.conv(eltwise8, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4c_branch2a#198");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4742088751518168e-05},{-inf},{inf}}, "res4c_branch2a_bias#105");
    auto bias_c31 = om.bias(conv31, biasWeights31, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights32 = om.constantInt(weightsData32,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.003781934268772602},{-0.49098676443099976},{0.47340646386146545}}, "res4c_branch2b_weights#107");
    auto conv32 = om.conv(bias_c31, weights32, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4c_branch2b#199");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.483111464040121e-05},{-inf},{inf}}, "res4c_branch2b_bias#108");
    auto bias_c32 = om.bias(conv32, biasWeights32, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights33 = om.constantInt(weightsData33,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.0038998068775981665},{-0.5584524273872375},{0.4359983205795288}}, "res4c_branch2c_weights#110");
    auto conv33 = om.conv(bias_c32, weights33, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4c_branch2c#200");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5293360775103793e-05},{-inf},{inf}}, "res4c_branch2c_bias#111");
    auto bias_c33 = om.bias(conv33, biasWeights33, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise9 = om.add(eltwise8,bias_c33, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4c/Relu#201");

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights34 = om.constantInt(weightsData34,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.0035950192250311375},{-0.4758487045764923},{0.4408811926841736}}, "res4d_branch2a_weights#114");
    auto conv34 = om.conv(eltwise9, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4d_branch2a#202");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4098114661464933e-05},{-inf},{inf}}, "res4d_branch2a_bias#115");
    auto bias_c34 = om.bias(conv34, biasWeights34, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights35 = om.constantInt(weightsData35,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.003885387210175395},{-0.5155961513519287},{0.4751776158809662}}, "res4d_branch2b_weights#117");
    auto conv35 = om.conv(bias_c34, weights35, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4d_branch2b#203");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.523681294202106e-05},{-inf},{inf}}, "res4d_branch2b_bias#118");
    auto bias_c35 = om.bias(conv35, biasWeights35, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData36 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights36 = om.constantInt(weightsData36,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003475428093224764},{-0.4302521049976349},{0.4559820592403412}}, "res4d_branch2c_weights#120");
    auto conv36 = om.conv(bias_c35, weights36, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4d_branch2c#204");

    std::vector<int64_t> biasWeightsData36 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights36 = om.constantInt(biasWeightsData36,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3629130080516916e-05},{-inf},{inf}}, "res4d_branch2c_bias#121");
    auto bias_c36 = om.bias(conv36, biasWeights36, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise10 = om.add(eltwise9,bias_c36, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4d/Relu#205");

    std::vector<int64_t> weightsData37 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights37 = om.constantInt(weightsData37,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0034418636932969093},{-0.420053631067276},{0.45762163400650024}}, "res4e_branch2a_weights#124");
    auto conv37 = om.conv(eltwise10, weights37, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4e_branch2a#206");

    std::vector<int64_t> biasWeightsData37 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights37 = om.constantInt(biasWeightsData37,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3497505278792232e-05},{-inf},{inf}}, "res4e_branch2a_bias#125");
    auto bias_c37 = om.bias(conv37, biasWeights37, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData38 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights38 = om.constantInt(weightsData38,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003783250693231821},{-0.479875773191452},{0.4848531484603882}}, "res4e_branch2b_weights#127");
    auto conv38 = om.conv(bias_c37, weights38, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4e_branch2b#207");

    std::vector<int64_t> biasWeightsData38 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights38 = om.constantInt(biasWeightsData38,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4836276932328474e-05},{-inf},{inf}}, "res4e_branch2b_bias#128");
    auto bias_c38 = om.bias(conv38, biasWeights38, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData39 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights39 = om.constantInt(weightsData39,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0035730148665606976},{-0.45972657203674316},{0.45139220356941223}}, "res4e_branch2c_weights#130");
    auto conv39 = om.conv(bias_c38, weights39, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4e_branch2c#208");

    std::vector<int64_t> biasWeightsData39 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights39 = om.constantInt(biasWeightsData39,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.401182271365542e-05},{-inf},{inf}}, "res4e_branch2c_bias#131");
    auto bias_c39 = om.bias(conv39, biasWeights39, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise11 = om.add(eltwise10,bias_c39, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4e/Relu#209");

    std::vector<int64_t> weightsData40 = mv::utils::generateSequence<int64_t> (1*1*1024*256);
    auto weights40 = om.constantInt(weightsData40,{1,1,1024,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0034057602752000093},{-0.4330636262893677},{0.4354052245616913}}, "res4f_branch2a_weights#134");
    auto conv40 = om.conv(eltwise11, weights40, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4f_branch2a#210");

    std::vector<int64_t> biasWeightsData40 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights40 = om.constantInt(biasWeightsData40,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3355922419577837e-05},{-inf},{inf}}, "res4f_branch2a_bias#135");
    auto bias_c40 = om.bias(conv40, biasWeights40, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData41 = mv::utils::generateSequence<int64_t> (3*3*256*256);
    auto weights41 = om.constantInt(weightsData41,{3,3,256,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.003915742039680481},{-0.44636479020118713},{0.5521494746208191}}, "res4f_branch2b_weights#137");
    auto conv41 = om.conv(bias_c40, weights41, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4f_branch2b#211");

    std::vector<int64_t> biasWeightsData41 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights41 = om.constantInt(biasWeightsData41,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.535585215606261e-05},{-inf},{inf}}, "res4f_branch2b_bias#138");
    auto bias_c41 = om.bias(conv41, biasWeights41, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData42 = mv::utils::generateSequence<int64_t> (1*1*256*1024);
    auto weights42 = om.constantInt(weightsData42,{1,1,256,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0037487009540200233},{-0.5017189979553223},{0.45419973134994507}}, "res4f_branch2c_weights#140");
    auto conv42 = om.conv(bias_c41, weights42, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4f_branch2c#212");

    std::vector<int64_t> biasWeightsData42 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights42 = om.constantInt(biasWeightsData42,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4700787687615957e-05},{-inf},{inf}}, "res4f_branch2c_bias#141");
    auto bias_c42 = om.bias(conv42, biasWeights42, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise12 = om.add(eltwise11,bias_c42, {{0},{0.003921568859368563},{0.0},{1.0}}, "res4f/Relu#213");

    std::vector<int64_t> weightsData43 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    auto weights43 = om.constantInt(weightsData43,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.003928038757294416},{-0.4819655418395996},{0.519684374332428}}, "res5a_branch1_weights#144");
    auto conv43 = om.conv(eltwise12, weights43, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a_branch1#214");

    std::vector<int64_t> biasWeightsData43 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights43 = om.constantInt(biasWeightsData43,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.540407356515061e-05},{-inf},{inf}}, "res5a_branch1_bias#145");
    auto bias_c43 = om.bias(conv43, biasWeights43, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData44 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    auto weights44 = om.constantInt(weightsData44,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{125},{0.003632510546594858},{-0.45575255155563354},{0.4705376625061035}}, "res5a_branch2a_weights#147");
    auto conv44 = om.conv(eltwise12, weights44, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a_branch2a#215");

    std::vector<int64_t> biasWeightsData44 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights44 = om.constantInt(biasWeightsData44,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4245139936974738e-05},{-inf},{inf}}, "res5a_branch2a_bias#148");
    auto bias_c44 = om.bias(conv44, biasWeights44, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData45 = mv::utils::generateSequence<int64_t> (1*1*512*256);
    auto weights45 = om.constantInt(weightsData45,{1,1,512,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{139},{0.0037927161902189255},{-0.5285225510597229},{0.43862009048461914}}, "res5a_branch2b_weights#150");
    auto conv45 = om.conv(bias_c44, weights45, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a_branch2b#216");

    std::vector<int64_t> biasWeightsData45 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights45 = om.constantInt(biasWeightsData45,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4873397049086634e-05},{-inf},{inf}}, "res5a_branch2b_bias#151");
    auto bias_c45 = om.bias(conv45, biasWeights45, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData46 = mv::utils::generateSequence<int64_t> (1*1*256*512);
    auto weights46 = om.constantInt(weightsData46,{1,1,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0036579601000994444},{-0.4496254324913025},{0.48315438628196716}}, "res5a_branch2c_weights#153");
    auto conv46 = om.conv(bias_c45, weights46, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a_branch2c#217");

    std::vector<int64_t> biasWeightsData46 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights46 = om.constantInt(biasWeightsData46,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4344941519084387e-05},{-inf},{inf}}, "res5a_branch2c_bias#154");
    auto bias_c46 = om.bias(conv46, biasWeights46, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise13 = om.add(bias_c43,bias_c46, {{0},{0.003921568859368563},{0.0},{1.0}}, "res5a/FakeQuantWithMinMaxArgs#218");
    
    om.output(eltwise13);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","GenerateWorkloads");
    unit.compilationDescriptor().remove("serialize","GenerateBlobKeembay");
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "MemoryHack", false);
    unit.compilationDescriptor().remove("finalize", "TensorGraphColoring");

    unit.initialize();
    unit.run();
   
    mv::DataModel dm(om);
    auto outflow = dm.getOutputFlow();
    uint64_t maxTopologicalCutValue = outflow->get<uint64_t>("MaxTopologicalCutValue");
    
    ASSERT_EQ(maxTopologicalCutValue, 1036288);
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