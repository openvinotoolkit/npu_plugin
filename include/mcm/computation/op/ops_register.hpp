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
        FullyConnected,
        MaxPool2D,
        AvgPool2D,
        Concat,
        ReLu,
        Softmax,
        Scale,
        BatchNorm,
        Add,
        Subtract,
        Muliply,
        Divide,
        Reshape,
        Bias

    };

}

#endif // OPS_REGISTER_