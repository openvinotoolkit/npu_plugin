#ifndef OPS_REGISTER_
#define OPS_REGISTER_

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
        ReLu,
        Softmax,
        Scale,
        BatchNorm,
        Add,
        Subtract,
        Multiply,
        Divide,
        Reshape,
        Bias

    };

}

#endif // OPS_REGISTER_