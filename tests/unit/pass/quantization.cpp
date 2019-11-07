#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

TEST(quantization, case_conv)
{
    double inf = std::numeric_limits<double>::infinity();

    //Test based on res2a_branch2a/quantized_model.tflite modeil in POC
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(64*64);
    auto weights0 = om.constantInt(weightsData,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{117},{0.002597350161522627},{-0.3044497072696686},{0.3578746020793915}}, "res2a_branch2a_weights#1");
    //    weights->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "res2a_branch2a#4");

    std::vector<int64_t> biasWeightsData0 = {000,-1702,000,000,000,16226,7740,14930,000,000,2931,-4815,000,000,3235,000,15412,7743,5898,3870,000,000,11536,000,-9447,-6232,-3706,14845,-4487,-3621,17741,1963,-9524,5490,000,000,000,000,-4179,000,000,16772,5611,000,000,000,-2581,693,000,8228,000,9119,000,-26908,12922,4479,000,-10064,000,000,000,7896,1087,2877};
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.0371373466332443e-05},{-inf},{inf}}, "res2a_branch2a_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));
    compDesc.setPassArg("GlobalConfigParams", "MemoryHack", false);
    compDesc.setPassArg("GlobalConfigParams", "MemoryHack", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    //ref data is based on result on POC test res2a_branch2a/quantized_model.tflite
    std::vector<int64_t> refData = {
        401408,16777215, 1427920640,49280,
        401472,16777215, 1427920640,47578,
        401536,16777215, 1427920640,49280,
        401600,16777215, 1427920640,49280,
        401664,16777215, 1427920640,49280,
        401728,16777215, 1427920640,65506,
        401792,16777215, 1427920640,57020,
        401856,16777215, 1427920640,64210,
        401920,16777215, 1427920640,49280,
        401984,16777215, 1427920640,49280,
        402048,16777215, 1427920640,52211,
        402112,16777215, 1427920640,44465,
        402176,16777215, 1427920640,49280,
        402240,16777215, 1427920640,49280,
        402304,16777215, 1427920640,52515,
        402368,16777215, 1427920640,49280,
        402432,16777215, 1427920640,64692,
        402496,16777215, 1427920640,57023,
        402560,16777215, 1427920640,55178,
        402624,16777215, 1427920640,53150,
        402688,16777215, 1427920640,49280,
        402752,16777215, 1427920640,49280,
        402816,16777215, 1427920640,60816,
        402880,16777215, 1427920640,49280,
        402944,16777215, 1427920640,39833,
        403008,16777215, 1427920640,43048,
        403072,16777215, 1427920640,45574,
        403136,16777215, 1427920640,64125,
        403200,16777215, 1427920640,44793,
        403264,16777215, 1427920640,45659,
        403328,16777215, 1427920640,67021,
        403392,16777215, 1427920640,51243,
        403456,16777215, 1427920640,39756,
        403520,16777215, 1427920640,54770,
        403584,16777215, 1427920640,49280,
        403648,16777215, 1427920640,49280,
        403712,16777215, 1427920640,49280,
        403776,16777215, 1427920640,49280,
        403840,16777215, 1427920640,45101,
        403904,16777215, 1427920640,49280,
        403968,16777215, 1427920640,49280,
        404032,16777215, 1427920640,66052,
        404096,16777215, 1427920640,54891,
        404160,16777215, 1427920640,49280,
        404224,16777215, 1427920640,49280,
        404288,16777215, 1427920640,49280,
        404352,16777215, 1427920640,46699,
        404416,16777215, 1427920640,49973,
        404480,16777215, 1427920640,49280,
        404544,16777215, 1427920640,57508,
        404608,16777215, 1427920640,49280,
        404672,16777215, 1427920640,58399,
        404736,16777215, 1427920640,49280,
        404800,16777215, 1427920640,22372,
        404864,16777215, 1427920640,62202,
        404928,16777215, 1427920640,53759,
        404992,16777215, 1427920640,49280,
        405056,16777215, 1427920640,39216,
        405120,16777215, 1427920640,49280,
        405184,16777215, 1427920640,49280,
        405248,16777215, 1427920640,49280,
        405312,16777215, 1427920640,57176,
        405376,16777215, 1427920640,50367,
        405440,16777215, 1427920640,52157
    };

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            auto weightTableName = "DMA"+dpuTask->getName()+"WeightsTable:0";
            auto inputs = dpuTask->getInputTensor();
            for (auto itr=inputs.begin(); itr != inputs.end(); itr++)
            {
                //if (itr-> dpuTask->getName()+"WeightsTable"
                if ((*itr)->getName() == weightTableName)
                {
                    auto weightsTableOp = om.getSourceOp(*itr);
                    weightsTableOp = weightsTableOp.leftmostParent();
                    auto weightsTable = weightsTableOp->getOutputTensor(0);
                    auto resData = weightsTable->getIntData();
                    for (unsigned i = 0; i < resData.size(); i+=4)
                    {
                        ASSERT_EQ(resData[i+2], refData[i+2]);
                        ASSERT_EQ(resData[i+3], refData[i+3]);
                    }
                }
            }
        }
    }
}
