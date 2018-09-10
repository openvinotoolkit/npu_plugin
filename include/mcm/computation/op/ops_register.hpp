#ifndef OPS_REGISTER_
#define OPS_REGISTER_

#include <map>
#include <string>

namespace mv
{

    enum class OpType
    {

        Input,
        Output,
        Constant,
        Conv2D,
        Conversion,
        MatMul,
        MaxPool2D,
        AvgPool2D,
        Concat,
        ReLU,
        PReLU,
        Softmax,
        Scale,
        BatchNorm,
        Add,
        Subtract,
        Multiply,
        Divide,
        Reshape,
        Bias,
        FullyConnected

    };

    const std::map<OpType, std::string> opsStrings
    {
        {OpType::Input, "input"},
        {OpType::Output, "output"},
        {OpType::Constant, "const"},
        {OpType::Conv2D, "conv2d"},
        {OpType::Conversion, "conversion"},
        {OpType::MatMul, "matmul"},
        {OpType::MaxPool2D, "maxpool2d"},
        {OpType::AvgPool2D, "avgpool2d"},
        {OpType::Concat, "concat"},
        {OpType::ReLU, "relu"},
        {OpType::PReLU, "prelu"},
        {OpType::Softmax, "softmax"},
        {OpType::Scale, "scale"},
        {OpType::BatchNorm, "batchnorm"},
        {OpType::Add, "add"},
        {OpType::Subtract, "subtract"},
        {OpType::Multiply, "multiply"},
        {OpType::Divide, "divide"},
        {OpType::Reshape, "reshape"},
        {OpType::Bias, "bias"},
        {OpType::FullyConnected, "fullyconnected"}
    };

    const std::map<std::string, OpType> opsStringsReversed
    {
        {"input", OpType::Input},
        {"output", OpType::Output},
        {"const", OpType::Constant},
        {"conv2d", OpType::Conv2D},
        {"conversion", OpType::Conversion},
        {"matmul", OpType::MatMul},
        {"maxpool2d", OpType::MaxPool2D},
        {"avgpool2d", OpType::AvgPool2D},
        {"concat", OpType::Concat},
        {"relu", OpType::ReLU},
        {"prelu", OpType::PReLU},
        {"softmax", OpType::Softmax},
        {"scale", OpType::Scale},
        {"batchnorm", OpType::BatchNorm},
        {"add", OpType::Add},
        {"subtract", OpType::Subtract},
        {"multiply", OpType::Multiply},
        {"divide", OpType::Divide},
        {"reshape", OpType::Reshape},
        {"bias", OpType::Bias},
        {"fullyconnected", OpType::FullyConnected}
    };

}

#endif // OPS_REGISTER_
