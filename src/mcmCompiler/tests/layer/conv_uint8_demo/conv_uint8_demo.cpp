#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("UInt8Demo");
    mv::OpModel& om = unit.model();

    //Input full of -0.5s
    auto input0 = om.input("input#170", {16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1.0/255.0},{},{}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (32*16*3*3, 255, 0); 
    //Wights first input channel of output channel full of 1s
    auto weights0 = om.constantInt("", weightsData0, {3,3,16,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    auto conv0 = om.conv("conv", input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    weights0->setQuantParams({{0},{1.0/255.0},{},{}});
    conv0->setQuantParams({{0},{144.0/255.0},{},{}});

    om.output("", conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
//    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize"); //So that we really use fp16

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
