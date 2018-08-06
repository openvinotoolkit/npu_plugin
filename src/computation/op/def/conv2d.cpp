#include "include/mcm/computation/op/def/conv2d.hpp"

mv::op::Conv2D::Conv2D(UnsignedVector2D stride, UnsignedVector4D padding, const string& name) :
ComputationOp(OpType::Conv2D, name),
KernelOp(OpType::Conv2D, stride, padding, name),
SinkOp(OpType::Conv2D, 2, name)
{
    addAttr("executable", AttrType::BoolType, true);
}

mv::op::Conv2D::Conv2D(mv::json::Value& obj) :
ComputationOp(obj),
KernelOp(obj),
SinkOp(obj)
{

}

mv::Tensor mv::op::Conv2D::getOutputDef(byte_type idx)
{

    if (idx > 0)
        return Tensor();

    if (!validOutputDef_())
        return Tensor();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    if (inputShape.ndims() != 3)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of incorrect shape " + inputShape.toString() + " of input");
        return Tensor();
    }
    
    if (weightsShape.ndims() != 4)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                "' because of incorrect shape " + weightsShape.toString() + " of weights");
        return Tensor();
    }

    if (inputShape[2] != weightsShape[2])
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
            "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[3])
            + ") and weights (" + Printable::toString(weightsShape[2]) + ")");
        return Tensor();
    }

    auto padding = getAttr("padding").getContent<UnsignedVector4D>();
    auto stride = getAttr("stride").getContent<UnsignedVector2D>();

    if (inputShape[0] + padding.e0 + padding.e1 < weightsShape[0])
    {
        logger_.log(Logger::MessageType::MessageError, 
            "Unable to define output tensor for '" + name_ + 
            "' because of filter kernel width (" + Printable::toString(weightsShape[0]) + 
            ") larger than padded input width (" + Printable::toString(inputShape[0] + padding.e0 + padding.e1) + ")");

        return Tensor();
    }

    if (inputShape[1] + padding.e2 + padding.e3 < weightsShape[1])
    {
        logger_.log(Logger::MessageType::MessageError, 
            "Unable to define output tensor for '" + name_ + 
            "' because of filter kernel height (" + Printable::toString(weightsShape[1]) + 
            ") larger than padded input height (" + Printable::toString(inputShape[1] + padding.e2 + padding.e3) + ")");

        return Tensor();
    }
    
    // Make sure that the result of subtract will not be negative
    Shape outputShape((inputShape[0] + padding.e0 + padding.e1 - weightsShape[0]) / stride.e0 + 1, (
        inputShape[1] + padding.e2 + padding.e3 - weightsShape[1]) / stride.e1 + 1, weightsShape[3]);

    return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

}

bool mv::op::Conv2D::isHardwarizeable(json::Object &TargetDescriptor)
{
    auto padding = getAttr("padding").getContent<UnsignedVector4D>();
    auto stride = getAttr("stride").getContent<UnsignedVector2D>();

    auto input = getInputTensor(0);
    auto inputShape = input->getShape();
    auto weights = getInputTensor(1);
    auto weightsShape = weights->getShape();

    // Check for supported padding
    if((padding.e0 != 0 && padding.e0 != weightsShape[0]/2) || (padding.e2 != 0 && padding.e2 != weightsShape[1]/2))
        return false;

    // Check for supported kernel sizes
    if(weightsShape[0] > 15 || weightsShape[1] > 15)
        return false;

    // Check for supported strides
    if(stride.e0 > 8 || stride.e1 > 8)
        return false;


    // Should handle dilation here

    // Should run optimizer for mode selection here

    return true;
}
