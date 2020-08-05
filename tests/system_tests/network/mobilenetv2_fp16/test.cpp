//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/op_model.hpp"

#include "mobilenetv2_fp16.hpp"

int main()
{
    const std::string binLocation = mv::utils::projectRootPath() +
        "/tests/system_tests/network/mobilenetv2_fp16/";
    mv::CompilationUnit compilationUnit = buildMobilenetV2_fp16(binLocation);
    mv::OpModel& opModel = compilationUnit.model();

    // Initialize and run the compilation unit.
    compilationUnit.loadCompilationDescriptor(mv::utils::projectRootPath() +
        "/config/compilation/release_kmb.json");
    compilationUnit.loadTargetDescriptor(mv::Target::ma2490);
    compilationUnit.initialize();
    compilationUnit.run();

}
