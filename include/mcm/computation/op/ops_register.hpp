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
        MatMul,
        MaxPool2D,
        AvgPool2D,
        Concat,
        ReLU,
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
        {OpType::MatMul, "matmul"},
        {OpType::MaxPool2D, "maxpool2d"},
        {OpType::AvgPool2D, "avgpool2d"},
        {OpType::Concat, "concat"},
        {OpType::ReLU, "relu"},
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

}

#endif // OPS_REGISTER_