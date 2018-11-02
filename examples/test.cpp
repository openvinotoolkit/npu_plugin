#include "include/mcm/computation/op/op_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "meta/include/mcm/recorded_compositional_model.hpp"

int main()
{
    //mv::op::OpRegistry::generateCompositionAPI();
    //mv::op::OpRegistry::generateRecordedCompositionAPI();

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    mv::OpModel model("model");
    mv::RecordedCompositionalModel om(model, "/home/smaciag/tmp/");
    auto input = om.input({32, 32, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto weightsData = mv::utils::generateSequence<double>(3 * 3 * 3 * 8);
    auto weights = om.constant(weightsData, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv = om.conv(input, weights, {2, 2}, {1, 1, 1, 1});
    om.output(conv);    

    return 0;
}