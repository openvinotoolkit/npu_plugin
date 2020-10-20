
// This Test validates the compiler and RT correctness for a network of type input->MaxPool->Eltwise (2nd input from 'Input' )->Conv
// The complexity in this network demonstrates the correctness of alignment of channels along the network layers

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t> (31104, 255, 0);

    const auto input0 = model.input("input0", {40, 24, 72, 1}, mv::DType("UInt8"), mv::Order("NHWC"));
    const auto pool0 = model.maxPool("", input0, {1, 1}, {1, 1,}, {0, 0, 0, 0}, true);
    const auto add0 = model.eltwise("", {input0, pool0}, "Add");
    const auto conv0_weights_0 = model.constantInt("conv0_weights0", weightsData, {3, 3, 72, 48}, mv::DType("UInt8"), mv::Order("NCHW"));
    const auto conv0 = model.conv("conv0", add0, conv0_weights_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto output = model.output("", conv0);

    input0->setQuantParams({{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}});
    pool0->setQuantParams({{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}});
    add0->setQuantParams({{0},{0.007843137718737},{0.000000000000000},{2.000000000000000},{0},{1}});
    conv0_weights_0->setQuantParams({{0},{0.003921568859369},{0},{1},{0},{1}});
    conv0->setQuantParams({{0},{0.070588239468642},{0.000000000000000},{18.000000000000000},{0},{1}});
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
