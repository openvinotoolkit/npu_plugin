#include "include/mcm/computation/op/def/drop_out.hpp"


mv::op::DropOut::DropOut(const std::string &name) :
ComputationOp(OpType::DropOut, name),
SinkOp(OpType::DropOut, 1, name),
SourceOp(OpType::DropOut, 1, name)
{
    set<bool>("executable", true);
}

mv::Tensor mv::op::DropOut::getOutputDef(std::size_t idx)
{

    // Will throw on error
    validOutputDef_(idx);

    auto input = getInputTensor(0);

   return Tensor(name_ + ":0", input->getShape(), input->getDType(), input->getOrder());

}
bool mv::op::DropOut::isHardwarizeable(json::Object&)
{
    return false;
}
