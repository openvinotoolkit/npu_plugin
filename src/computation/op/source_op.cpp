#include "include/fathom/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(OpType opType, const string &name) :
ComputationOp(opType, name)
{
    addAttr("output", AttrType::StringType, name_ + ":out");
}

mv::SourceOp::~SourceOp()
{

}

bool mv::SourceOp::setOutput(DataContext::TensorIterator &tensor)
{
    output_ = tensor;
    logger_.log(Logger::MessageType::MessageDebug, "Set output for " + toString() + " as " + tensor->toString());
    return true;
}

mv::DataContext::TensorIterator mv::SourceOp::getOutput()
{
    return output_;
}

mv::string mv::SourceOp::getOutputName() const
{
    return getAttr("output").getContent<string>();
}