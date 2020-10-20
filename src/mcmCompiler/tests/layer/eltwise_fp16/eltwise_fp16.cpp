#include "include/mcm/compiler/compilation_unit.hpp"

int main()
{
    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    //Input
    auto input0 = om.input("input0", {7,7,1024,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    auto input1 = om.input("input1", {7,7,1024,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1.0},{},{}});
    input1->setQuantParams({{0},{1.0},{},{}});

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(input1);

    // Eltwise, to be replaced with eltwiseFP16 in replacement pass
    auto eltwise0 = om.eltwise("eltwise0", inputs, "Add");
    eltwise0->setQuantParams({{0},{1.0},{},{}});

    // Output
    auto output0 = om.output("", eltwise0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
