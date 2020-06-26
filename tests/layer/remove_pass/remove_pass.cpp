#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("RemovePass");
    mv::OpModel& om = unit.model();

    //Input full of 1s
    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0/255.0},{},{}}, "input#170");

    mv::Shape beginShape = mv::Shape({0,0,0,0});
    mv::Shape endShape = mv::Shape({16,16,16,1});

    auto slice0 = om.slice(input0, beginShape, endShape, {{0},{1.0},{},{}}, "slice");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (32*16*3*3, 255, 0); 
    //Wights first input channel of output channel full of 1s
    auto weights0 = om.constantInt(weightsData0,{3,3,16,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0/255.0},{},{}});
    auto conv0 = om.conv(slice0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{144.0/255.0},{},{}} , "conv");

    auto dropout = om.dropout(conv0);

    om.output(dropout);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
