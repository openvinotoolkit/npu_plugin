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

    auto input1 = test_cm.input({16, 16, 16}, mv::DType("Float16"), mv::Order("CHW"));
    auto input1dmaIN = test_cm.dMATask(input1, mv::DmaDirectionEnum::DDR2CMX);
    test_cm.deAllocate(input1dmaIN);
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto dmaINweights1 = test_cm.dMATask(weights1, mv::DmaDirectionEnum::DDR2CMX);
    test_cm.deAllocate(dmaINweights1);
    auto dpuconv1 = test_cm.dPUTaskConv({input1dmaIN, dmaINweights1}, {1,1}, {1,1,1,1});
    auto dmaOutput = test_cm.dMATask(dpuconv1, mv::DmaDirectionEnum::CMX2DDR);
    test_cm.output(dmaOutput);


    auto inputOp = test_cm.getSourceOp(input1);
    auto input1dmaINOp = test_cm.getSourceOp(input1dmaIN);
    auto input1dmaOutOp = test_cm.getOp("DeAllocate_0");
    auto weightsOp = test_cm.getSourceOp(weights1);
    auto dmaINweights1Op = test_cm.getSourceOp(dmaINweights1);
    auto dmaOUTWeights1Op = test_cm.getOp("DeAllocate_1");
    auto dpuconv1Op = test_cm.getSourceOp(dpuconv1);
    auto dmaOutputOp = test_cm.getSourceOp(dmaOutput);
    auto outputOp = test_cm.getOp("Output_0");

    std::string outputName("dpu_task");
    cm.defineFlow(inputOp, input1dmaINOp);
    cm.defineFlow(inputOp, dmaINweights1Op);

    cm.defineFlow(input1dmaINOp, dpuconv1Op);
    cm.defineFlow(dmaINweights1Op, dpuconv1Op);

    cm.defineFlow(input1dmaINOp, input1dmaOutOp);
    cm.defineFlow(dmaINweights1Op, dmaOUTWeights1Op);

    cm.defineFlow(input1dmaOutOp, dmaOutputOp);
    cm.defineFlow(dmaOUTWeights1Op, dmaOutputOp);
    cm.defineFlow(dpuconv1Op, dmaOutputOp);
    cm.defineFlow(dpuconv1Op, input1dmaOutOp);
    cm.defineFlow(dpuconv1Op, dmaOUTWeights1Op);


    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.run();

    system("dot -Tsvg dpu_task.dot -o dpu_task.png");
    system("dot -Tsvg dpu_task_adapt.dot -o dpu_task_adapt.png");

}
