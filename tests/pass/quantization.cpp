#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

TEST(quantization, case_conv)
{
    //Test based on res2a_branch2a/quantized_model.tflite modeil in POC
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::QuantizationParams inputQuantParams({128}, {0.00784314}, {0}, {1});
    auto input = om.input({56, 56, 64, 1}, mv::DType("UInt8"), mv::Order("NWHC"), inputQuantParams);

    //    input->set<mv::QuantizationParams>("quantParams", inputQuantParams);
    auto testShape = input->getShape();
    //EC: output defs are deduced from inputs

    mv::QuantizationParams weightsQuantParams({120}, {0.0028294341}, {0}, {1});
    mv::QuantizationParams biasQuantParams({0}, {2.219164e-05}, {0}, {1});

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(64*64);
    auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order(mv::Order::getColMajorID(4)), weightsQuantParams, "weights");
    //    weights->set<mv::QuantizationParams>("quantParams", weightsQuantParams);
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, inputQuantParams);
    std::vector<int64_t> biasesData = {
    13559,  17916,  18802,   2546,   2108,  -6720,  11957,   6745,   7859,   4116,
    3767,  11175,    559,  14635,  10865,  -5677,   6943,    996,   -427,  -3778,
    8863,   6461,   7315,  10601,  -2944,  12207,  -2114,  -1911,   1976,  -2687,
    19133,  -4836,   6276,   -841,  14684,  13039,  -5271,  13416, -12849, -13113,
    4303,  15551,   3331,    108,   7871,   2019,   5212,   8503,  -3542,   2456,
    7652,   8379,  -9771,   -866,  -8929,   3194, -12861,  -2842,   3623,   8634,
    1904,    778,   3990,   6220
    };

    auto constantBias = om.constantInt(biasesData, {64}, mv::DType("UInt8"), mv::Order("W"));
    auto bias = om.bias(conv, constantBias, biasQuantParams);
    auto output = om.output(bias);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    //ref data is based on result on POC test res2a_branch2a/quantized_model.tflite
    std::vector<double> refData = {
    0 , 0 , 1555502848 , 58797,
    0 , 0 , 1555502848 , 63154,
    0 , 0 , 1555502848 , 64040,
    0 , 0 , 1555502848 , 47784,
    0 , 0 , 1555502848 , 47346,
    0 , 0 , 1555502848 , 38518,
    0 , 0 , 1555502848 , 57195,
    0 , 0 , 1555502848 , 51983,
    0 , 0 , 1555502848 , 53097,
    0 , 0 , 1555502848 , 49354,
    0 , 0 , 1555502848 , 49005,
    0 , 0 , 1555502848 , 56413,
    0 , 0 , 1555502848 , 45797,
    0 , 0 , 1555502848 , 59873,
    0 , 0 , 1555502848 , 56103,
    0 , 0 , 1555502848 , 39561,
    0 , 0 , 1555502848 , 52181,
    0 , 0 , 1555502848 , 46234,
    0 , 0 , 1555502848 , 44811,
    0 , 0 , 1555502848 , 41460,
    0 , 0 , 1555502848 , 54101,
    0 , 0 , 1555502848 , 51699,
    0 , 0 , 1555502848 , 52553,
    0 , 0 , 1555502848 , 55839,
    0 , 0 , 1555502848 , 42294,
    0 , 0 , 1555502848 , 57445,
    0 , 0 , 1555502848 , 43124,
    0 , 0 , 1555502848 , 43327,
    0 , 0 , 1555502848 , 47214,
    0 , 0 , 1555502848 , 42551,
    0 , 0 , 1555502848 , 64371,
    0 , 0 , 1555502848 , 40402,
    0 , 0 , 1555502848 , 51514,
    0 , 0 , 1555502848 , 44397,
    0 , 0 , 1555502848 , 59922,
    0 , 0 , 1555502848 , 58277,
    0 , 0 , 1555502848 , 39967,
    0 , 0 , 1555502848 , 58654,
    0 , 0 , 1555502848 , 32389,
    0 , 0 , 1555502848 , 32125,
    0 , 0 , 1555502848 , 49541,
    0 , 0 , 1555502848 , 60789,
    0 , 0 , 1555502848 , 48569,
    0 , 0 , 1555502848 , 45346,
    0 , 0 , 1555502848 , 53109,
    0 , 0 , 1555502848 , 47257,
    0 , 0 , 1555502848 , 50450,
    0 , 0 , 1555502848 , 53741,
    0 , 0 , 1555502848 , 41696,
    0 , 0 , 1555502848 , 47694,
    0 , 0 , 1555502848 , 52890,
    0 , 0 , 1555502848 , 53617,
    0 , 0 , 1555502848 , 35467,
    0 , 0 , 1555502848 , 44372,
    0 , 0 , 1555502848 , 36309,
    0 , 0 , 1555502848 , 48432,
    0 , 0 , 1555502848 , 32377,
    0 , 0 , 1555502848 , 42396,
    0 , 0 , 1555502848 , 48861,
    0 , 0 , 1555502848 , 53872,
    0 , 0 , 1555502848 , 47142,
    0 , 0 , 1555502848 , 46016,
    0 , 0 , 1555502848 , 49228,
    0 , 0 , 1555502848 , 51458
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
                    //this test only checks what is filled in populateWeightsTablesActivationAndBias function
                    //data_ptr and sparsity_ptr are not set in reference received from POC compiler
                    for (unsigned i = 0; i < resData.size(); i+=4)
                    {
                        resData[i] = 0;
                        resData[i + 1] = 0;
                    }
                    for (unsigned i = 0; i < resData.size(); i+=4)
                    {
                        ASSERT_FLOAT_EQ(resData[i+2], refData[i+2]);
                        ASSERT_FLOAT_EQ(resData[i+3], refData[i+3]);
                    }
                }
            }
        }
    }
}
