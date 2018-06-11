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
        MaxPool2D,
        Concat

    };

}

#endif // OPS_REGISTER_