#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/target/keembay/tasks/nce2_task_api.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& test_cm = unit.model();
    mv::ControlModel cm(test_cm);

    auto input1 = test_cm.input({225, 225, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto input1dmaIN = mv::createDMATask(test_cm, input1, mv::DmaDirection(mv::DmaDirectionEnum::DDR2CMX));
    mv::createBarrierTask(test_cm);
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DTypeType::Float16, mv::Order("NCWH"));
    auto weights1dmaIN = mv::createDMATask(test_cm, weights1, mv::DmaDirection(mv::DmaDirectionEnum::DDR2CMX));
    auto conv1 = test_cm.conv(input1dmaIN, weights1dmaIN, {2, 2}, {0, 0, 0, 0}, 1);
    auto conv1_dpu = mv::createDPUTask(test_cm, test_cm.getSourceOp(conv1));
    test_cm.removeOp(test_cm.getSourceOp(conv1));

    auto pool1 = test_cm.maxPool(conv1_dpu, {3, 3}, {1, 1}, {1, 1});
    auto pool1_dpu = mv::createDPUTask(test_cm, test_cm.getSourceOp(pool1));
    test_cm.removeOp(test_cm.getSourceOp(pool1));
    auto outputdmaout = mv::createDMATask(test_cm, pool1_dpu, mv::DmaDirection(mv::DmaDirectionEnum::CMX2DDR));
    auto output = test_cm.output(outputdmaout);

    //Defining ControlFlows for the graph
    //First, we get all the ops
    auto input1Op = test_cm.getSourceOp(input1);
    auto barrierTask = test_cm.getOp("BarrierTask_0");
    auto input1dmaINOp = test_cm.getSourceOp(input1dmaIN);
    auto weights1Op = test_cm.getSourceOp(weights1);
    auto weights1dmaINOp = test_cm.getSourceOp(weights1dmaIN);
    auto conv1_dpuOp = test_cm.getSourceOp(conv1_dpu);
    auto pool1_dpuOp = test_cm.getSourceOp(pool1_dpu);
    auto outputdmaoutOp = test_cm.getSourceOp(outputdmaout);
    auto outputOp = test_cm.getOp("Output_0");

    cm.defineFlow(input1Op, barrierTask);
    cm.defineFlow(barrierTask, input1dmaINOp);
    cm.defineFlow(input1dmaINOp, weights1Op);
    cm.defineFlow(weights1Op, weights1dmaINOp);
    cm.defineFlow(weights1dmaINOp, conv1_dpuOp);
    cm.defineFlow(conv1_dpuOp, pool1_dpuOp);
    cm.defineFlow(pool1_dpuOp, outputdmaoutOp);
    cm.defineFlow(outputdmaoutOp, outputOp);

    std::string outputName = "dpu_task";

    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;

    unit.loadTargetDescriptor(mv::Target::keembay);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.run();

    system("dot -Tsvg dpu_task.dot -o dpu_task.svg");
}
