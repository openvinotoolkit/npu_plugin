#include "include/fathom/computation/op/source_op.hpp"

mv::SourceOp::SourceOp(const Logger &logger, const string &opType, const string &name) :
ComputationOp(logger, opType, name)
{
    addAttr("output", AttrType::StringType, name_ + ":out");
}

mv::SourceOp::~SourceOp()
{

}

bool mv::SourceOp::setOutput(TensorContext::UnpopulatedTensorIterator &tensor)
{
    output_ = tensor;
    logger_.log(Logger::MessageType::MessageDebug, "Set output for " + toString() + " as " + tensor->toString());
    return true;
}

mv::TensorContext::UnpopulatedTensorIterator mv::SourceOp::getOutput()
{
    return output_;
}

mv::string mv::SourceOp::getOutputName() const
{
    return getAttr("output").getContent<string>();
}