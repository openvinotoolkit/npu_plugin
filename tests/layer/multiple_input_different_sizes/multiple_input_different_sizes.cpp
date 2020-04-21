//This file is the parsed network which is created through python.

// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"

void build_pySwigCU(mv::OpModel& model)
{
    using namespace mv;

    static const auto inf = std::numeric_limits<double>::infinity();

    const auto Parameter_0_0 = model.input({64, 64, 3, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "Parameter_0");
    const auto Parameter_1_0 = model.input({64, 64, 3, 1}, mv::DType("UInt8"), mv::Order("NHWC"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "Parameter_1");
    const auto Concat_0 = model.concat({Parameter_0_0, Parameter_1_0}, "C", mv::DType("Default"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "Concat_0");
    //const auto Add_2_0 = model.eltwise({Parameter_0_0, Parameter_1_0}, "Add", mv::DType("Default"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "Add_2");
    const auto output = model.output(Concat_0);

}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_pySwigCU(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
