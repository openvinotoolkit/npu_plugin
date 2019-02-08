#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& test_cm = unit.model();
    mv::ControlModel cm(test_cm);

    auto input1 = test_cm.input({225, 225, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto input1dmaIN = test_cm.dMATask(input1, mv::DmaDirectionEnum::DDR2CMX);
    test_cm.deAllocate(input1dmaIN);
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto dmaINweights1 = test_cm.dMATask(weights1, mv::DmaDirectionEnum::DDR2CMX);
    test_cm.deAllocate(dmaINweights1);
    auto barrier_task_0 = test_cm.barrierTask( input1dmaIN, 0,0,2,1,-1, "barrierTask0" );
    auto dpuconv1 = test_cm.dPUTaskConv({input1dmaIN, dmaINweights1}, {2,2}, {0,0,0,0});
    auto barrier_task_1 = test_cm.barrierTask( dpuconv1, 0,1,1,1,-1, "barrierTask1" );
    auto dmaOutput = test_cm.dMATask(dpuconv1, mv::DmaDirectionEnum::CMX2DDR);
    test_cm.output(dmaOutput);


    auto inputOp = test_cm.getSourceOp(input1);
    auto input1dmaINOp = test_cm.getSourceOp(input1dmaIN);
    auto input1dmaOutOp = test_cm.getOp("DeAllocate_0");
    auto weightsOp = test_cm.getSourceOp(weights1);
    auto dmaINweights1Op = test_cm.getSourceOp(dmaINweights1);
    auto barrier0_Op = test_cm.getSourceOp(barrier_task_0);
    auto dmaOUTWeights1Op = test_cm.getOp("DeAllocate_1");
    auto dpuconv1Op = test_cm.getSourceOp(dpuconv1);
    auto barrier1_Op = test_cm.getSourceOp(barrier_task_1);
    auto dmaOutputOp = test_cm.getSourceOp(dmaOutput);
    auto outputOp = test_cm.getOp("Output_0");

    std::string outputName("dpu_task");
    cm.defineFlow(inputOp, input1dmaINOp);
    cm.defineFlow(inputOp, dmaINweights1Op);

//    cm.defineFlow(input1dmaINOp, dpuconv1Op);
    cm.defineFlow(input1dmaINOp, barrier0_Op);
//    cm.defineFlow(dmaINweights1Op, dpuconv1Op);
    cm.defineFlow(dmaINweights1Op, barrier0_Op);
    cm.defineFlow(barrier0_Op, dpuconv1Op);

    cm.defineFlow(input1dmaINOp, input1dmaOutOp);
    cm.defineFlow(dmaINweights1Op, dmaOUTWeights1Op);

    cm.defineFlow(input1dmaOutOp, dmaOutputOp);
    cm.defineFlow(dmaOUTWeights1Op, dmaOutputOp);
    cm.defineFlow(barrier1_Op, dmaOutputOp);
    cm.defineFlow(dpuconv1Op, barrier1_Op);
    cm.defineFlow(dpuconv1Op, input1dmaOutOp);
    cm.defineFlow(dpuconv1Op, dmaOUTWeights1Op);


    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.run();

    system("dot -Tsvg dpu_task.dot -o dpu_task.png");
    system("dot -Tsvg dpu_task_final.dot -o dpu_task_final.png");

}
