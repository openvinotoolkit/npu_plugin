#include "include/mcm/computation/op/def/concat.hpp"

mv::op::Concat::Concat(const std::string &name) :
ComputationOp(OpType::Concat, name),
SinkOp(OpType::Concat, 2, name),
SourceOp(OpType::Concat, 1, name)
{
    set<bool>("executable", true);
    set<int>("axis", 2);
}

mv::Tensor mv::op::Concat::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape();

    if (input0Shape.ndims() != 3)
        throw(OpError(*this, "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
            " has " + std::to_string(input0Shape.ndims())));

    std::size_t lastDim = input0Shape[2];

    for (std::size_t i = 1; i < inputSlots(); ++i)
    {
        auto inputShape = getInputTensor(i)->getShape();
        if (inputShape.ndims() != 3)
            throw(OpError(*this, "Invalid shape of the input tensor (input " + std::to_string(i) + ") - must have a dimensionality of 3, "
                " has " + std::to_string(inputShape.ndims())));

        // TODO: based on concat axis, the other dimensions should match
        if (inputShape[0] != input0Shape[0] || inputShape[1] != input0Shape[1])
            throw(OpError(*this, "Invalid shape of the input tensor (input " + std::to_string(i) + ") - inconsistent with the dimension of "
                " the first input (input 0) "));

        lastDim += inputShape[2];

    }

    return Tensor(name_ + ":0", {input0Shape[0], input0Shape[1], lastDim}, input0->getDType(), input0->getOrder());

}

bool mv::op::Concat::isHardwarizeable(json::Object&)
{
    return false;
}
