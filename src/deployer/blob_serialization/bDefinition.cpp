#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include <stdio.h>

namespace mv
{
    Blob_Op_Definition::Blob_Op_Definition()
    {

    }

    Blob_Op_Definition::Blob_Op_Definition(OpType o)
    {

        // Number of Inputs

        this->number_of_inputs = -1;
        switch(o)
        {
            case OpType::Add:
            case OpType::Multiply:
            case OpType::Scale:
                this->number_of_inputs = 2;
                break;
            case OpType::Conv2D:
            case OpType::FullyConnected:
            case OpType::AvgPool2D:
            case OpType::MaxPool2D:
            case OpType::Softmax:
            case OpType::ReLU:
                this->number_of_inputs = 1;
                break;
            case OpType::Output:
            case OpType::Input:
                this->number_of_inputs = 0;
                break;
            default:
                printf("No Entry in 'numberOfInputs' for OpType #%i\n", (int)o);
                assert(0);
        }
    }
}