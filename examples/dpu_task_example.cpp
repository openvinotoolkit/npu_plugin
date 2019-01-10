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

    auto input1 = test_cm.input({225, 225, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto input1dmaIN = mv::createDMATask(test_cm, input1, mv::DmaDirection(mv::DmaDirectionEnum::DDR2CMX));
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

    std::string outputName = "dpu_task";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = outputName + ".blob";
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string(outputName + ".dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().disablePass(mv::PassGenre::Adaptation);
    unit.run();

    system("dot -Tsvg dpu_task.dot -o dpu_task.svg");
    system("dot -Tsvg dpu_task_adapt.dot -o dpu_task_adapt.svg");
    system("dot -Tsvg dpu_task_final.dot -o dpu_task_final.svg");
}
