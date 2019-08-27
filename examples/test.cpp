#include "include/mcm/computation/op/op_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "meta/include/mcm/recorded_compositional_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"


int main()
{
    
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

   auto input0 = om.input({112,122,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (7*7*3*64);
    auto weights0 = om.constantInt(weightsData0,{7,7,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.002871257718652487},{-0.32948583364486694},{0.40268489718437195}}, "conv1/conv1_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {2, 3, 2, 3}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1/conv1#4");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2519669073517434e-05},{-inf},{inf}}, "conv1/conv1_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    om.output(bias_c0);

  
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490-resnet50-reduced-streaming.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    return 0;
}
