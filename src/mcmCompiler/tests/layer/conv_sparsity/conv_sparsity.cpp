#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("weightSparsity");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("input#170", {64,64,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{0.00392156862745098},{},{}});

    auto pool0 = om.maxPool("pool", input0, {1,1}, {1,1}, {0,0,0,0}, true);
    pool0->setQuantParams({{0},{0.00392156862745098},{},{}});

    //NOTE: The idea here is that the first 64 elements of weights will be 1 and all the rest 0s. Doing that the whole output
    //tensor is expected to be full of 1s. Enable weight sparsity and validate...
    //In SOK, if the last cluster for example was taking full weight tensor of zeros, runtime would hang, runtime needs
    //to handle that case if it hits us in the future...
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (64, 255, 0);
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (4032, 0, 0);
    weightsData0.insert(weightsData0.end(), weightsData1.begin(), weightsData1.end());

    auto weights0 = om.constantInt("", weightsData0,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getRowMajorID(4));
    auto conv0 = om.conv("conv", pool0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    weights0->setQuantParams({{0},{0.00392156862745098},{},{}});
    conv0->setQuantParams({{0},{0.00392156862745098},{},{}});

    om.output("", conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
