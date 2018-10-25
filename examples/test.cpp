#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{


    mv::OpModel m1("m1");

    mv::Op in(m1, "Input", "input1", {},
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

    mv::Op out(m1, "Output", "output1", {conv.getOutputTensor(0)});

    std::cout << conv.toString() << std::endl;
    std::cout << conv.getInputTensor("data")->getName() << std::endl;
    std::cout << conv.getOutputTensor(0)->toString() << std::endl;

    std::cout << out.getInputTensor(0)->toString() << std::endl;

    std::cout << weights.getOutputTensor(0)->toString() << std::endl;
    std::cout << weights.getOutputTensor(0)->toJSON().stringifyPretty() << std::endl;
    std::cout << weights.getOutputTensor(0)->getData()[0] << std::endl;

    std::cout << weights.toJSON().stringifyPretty() << std::endl;

}