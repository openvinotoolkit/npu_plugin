#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("DepthwiseConvolution");
    mv::CompositionalModel& test_cm = unit.model();

    auto input = test_cm.input({225, 225, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*1);
    auto weights1 = test_cm.constant(weightsData, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    auto conv = test_cm.depthwiseConv(input, weights1, {4, 4}, {1, 1, 1, 1});
    auto output = test_cm.output(conv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    //compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng keembay_adapt_model.dot -o keembay_adapt_model.png");
    system("dot -Tpng dma_model.dot -o dma_model.png");
    system("dot -Tpng control_model.dot -o control_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}
