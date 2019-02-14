#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("RuntimeModelExample");
    mv::OpModel& test_cm = unit.model();
    mv::ControlModel cm(test_cm);

    unit.loadCompilationDescriptor(mv::utils::projectRootPath()+"/config/compilation/ma2490.json");

    auto input1 = test_cm.input({225, 225, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto input1dmaIN = test_cm.dMATask(input1, mv::DmaDirectionEnum::DDR2CMX);
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto dmaINweights1 = test_cm.dMATask(weights1, mv::DmaDirectionEnum::DDR2CMX);
    auto dpuconv1 = test_cm.dPUTaskConv({input1dmaIN, dmaINweights1}, {2,2}, {0,0,0,0});
    auto dmaOutput = test_cm.dMATask(dpuconv1, mv::DmaDirectionEnum::CMX2DDR);
    test_cm.output(dmaOutput);

    std::string outputName("dpu_task");

    mv::CompilationDescriptor& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "output", std::string(outputName + ".dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);
    compDesc.setPassArg("GenerateBlobKeembay", "Output", std::string(outputName + ".blob"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tsvg dpu_task.dot -o dpu_task.png");
    system("dot -Tsvg dpu_task_final.dot -o dpu_task_final.png");
}
