#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/print_info_pass.hpp"

int main()
{
    mv::CompilationUnit unit("RegionYolo");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({13,13,125,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");
    // Define Params
    unsigned coords = 4;
    unsigned classes = 20;
    bool do_softmax = true;
    unsigned num = 5;
    std::vector<unsigned> mask;
    auto regionyolo0 = om.regionYolo(input0, coords, classes, do_softmax, num, mask, mv::DType("Float16"));
    om.output(regionyolo0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Silent")));
    unit.compilationDescriptor().addToGroup("serialize", "PrintInfo", "Singular", false);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
