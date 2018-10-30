#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/api/compositional_model.hpp"

int main()
{

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    mv::OpModel m1("m1");

    /*mv::Op in(m1, "Input", "input1", {},
        {
            { "shape", mv::Shape({32, 32, 3}) },
            { "dType", mv::DType(mv::DTypeType::Float16) },
            { "order", mv::Order(mv::OrderType::ColumnMajor) }
        }
    );

    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);

    mv::Op weights(m1, "Constant", "constant1", {},
        {
            { "shape", mv::Shape({3, 3, 3, 8}) },
            { "dType", mv::DType(mv::DTypeType::Float16) },
            { "order", mv::Order(mv::OrderType::ColumnMajor) },
            { "data", weights1Data }
        }
    );

    mv::Op conv(m1, "Conv", "conv1", {in.getOutputTensor(0), weights.getOutputTensor(0)},
        {
            { "padding", std::array<unsigned short, 4>({1, 1, 1, 1}) },
            { "stride", std::array<unsigned short, 2>({2, 2}) }
        }
    );

    mv::Op out(m1, "Output", "output1", {conv.getOutputTensor(0)});*/

    

    /*auto in = m1.defineOp("Input", {},
        {
            { "shape", mv::Shape({32, 32, 3}) },
            { "dType", mv::DType(mv::DTypeType::Float16) },
            { "order", mv::Order(mv::OrderType::ColumnMajor) }
        }
    );

    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);

    auto weights = m1.defineOp("Constant", {},
        {
            { "shape", mv::Shape({3, 3, 3, 8}) },
            { "dType", mv::DType(mv::DTypeType::Float16) },
            { "order", mv::Order(mv::OrderType::ColumnMajor) },
            { "data", weights1Data }
        }
    );

    auto conv = m1.defineOp("Conv", {in, weights},
        {
            { "padding", std::array<unsigned short, 4>({1, 1, 1, 1}) },
            { "stride", std::array<unsigned short, 2>({2, 2}) }
        }
    );

    m1.defineOp("Output", {conv});*/

    mv::op::OpRegistry::generateCompositionAPI();

    /*mv::OpModel om("testModel");
    mv::CompositionalModel model(om);

    auto input = model.input(mv::DTypeType::Float16, mv::OrderType::ColumnMajor, {32, 32, 3});
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    auto weights = model.constant(mv::DTypeType::Float16, weights1Data, mv::OrderType::ColumnMajor, {3, 3, 3, 8});
    auto conv = model.conv(input, weights, {1, 1, 1, 1}, {2, 2});
    auto output = model.output(conv);*/

}