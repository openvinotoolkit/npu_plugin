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

    auto input1 = test_cm.input({225, 225, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DTypeType::Float16, mv::Order("NCWH"));
    std::vector<std::pair<std::string, mv::Attribute>> convDpuAttr;
    convDpuAttr.push_back(std::make_pair("taskOp", std::string("Conv")));
    convDpuAttr.push_back(std::make_pair("stride", std::array<unsigned short, 2>({2,2})));
    convDpuAttr.push_back(std::make_pair("padding", std::array<unsigned short, 4>({0,0,0,0})));
    convDpuAttr.push_back(std::make_pair("dilationFactor", (unsigned)1));

    auto dpuconv1 = test_cm.dPUTask(std::vector<mv::Data::TensorIterator>({input1, weights1}), convDpuAttr);
    //auto conv1 = test_cm.conv(input1, weights1, {2, 2}, {0, 0, 0, 0}, 1);

    std::vector<std::pair<std::string, mv::Attribute>> poolDpuAttr;
    poolDpuAttr.push_back(std::make_pair("taskOp", std::string("MaxPool")));
    poolDpuAttr.push_back(std::make_pair("kSize", std::array<unsigned short, 2>({3,3})));
    poolDpuAttr.push_back(std::make_pair("stride", std::array<unsigned short, 2>({3,3})));
    poolDpuAttr.push_back(std::make_pair("padding", std::array<unsigned short, 4>({0,0,0,0})));
    auto dpupool1 = test_cm.dPUTask({dpuconv1}, poolDpuAttr);
    //auto pool1 = test_cm.maxPool(conv1, {3, 3}, {1, 1}, {1, 1});
    //auto output = test_cm.output(pool1);
    auto output = test_cm.output(dpupool1);

    std::string outputName("dpu_task");

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
