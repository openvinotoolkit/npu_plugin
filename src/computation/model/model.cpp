#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;

mv::ComputationModel::ComputationModel(Logger &logger) : 
logger_(logger)
{

}

mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
logger_(getLogger(verboseLevel, logTime))
{

}

const mv::OpListIterator mv::ComputationModel::input(const Shape &shape, DType dType, Order order, const string &name)
{

    string inputName = name;

    if (inputName.empty())
        inputName = "0";

    //allocator::owner_ptr<Input> inputPtr = ;
    //auto inputOpPtr = cast_pointer<ComputationOp>(inputPtr);

    input_ = ops_graph.node_insert(allocator_.make_owner<Input>(logger_, inputName, shape, dType, order));
    //input_ = ops_graph.node_insert(Input(logger_, inputName, shape, dType, order));

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + input_->toString());

    return input_;

}

const mv::OpListIterator mv::ComputationModel::output(OpListIterator &predecessor, const string &name)
{

    string outputName = name;

    if (outputName.empty())
        outputName = "0";

    //auto outputShape = predecessor->getOutputShape();
    //auto outputDType = predecessor->getDType();
    //auto outputOrder = predecessor->getOrder();
    //VariableTensor inputTensor(logger_, name + "_output", outputShape, outputDType, outputOrder);
    auto inTensor = predecessor->getOutput();
    output_ = ops_graph.node_insert(predecessor, allocator_.make_owner<Output>(logger_, outputName, inTensor), inTensor);
    //output_ = ops_graph.node_insert(predecessor, Output(logger_, outputName, inTensor), inTensor);

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + output_->toString());

    return output_;

}

mv::OpListIterator mv::ComputationModel::convolutional(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, const string &name)
{

    string convName = name;

    if (convName.empty())
        convName = "0";

    auto inputShape = predecessor->getOutputShape();
    //auto convDType = predecessor->getDType();
    //auto convOrder = predecessor->getOrder();

    Shape convShape(inputShape[0], inputShape[1] / strideX, inputShape[2] / strideY, weights.getShape()[2]);

    //VariableTensor inputTensor(logger_, name + "_input", inputShape, convDType, convOrder);

    auto inTensor = predecessor->getOutput();

    //OpListIterator conv = ops_graph.node_insert(predecessor, Conv(logger_, convName, inTensor, weights, strideX, strideY), inTensor);
    OpListIterator conv = ops_graph.node_insert(predecessor, allocator_.make_owner<Conv>(logger_, convName, inTensor, weights, strideX, strideY), inTensor);

    /*conv->addAttr("weigths", ComputationElement::TensorType, ConstantModelTensor(logger_, convName + "_weights", weights));
    conv->addAttr("strideX", ComputationElement::ByteType, strideX);
    conv->addAttr("strideY", ComputationElement::ByteType, strideY);*/

    logger_.log(Logger::MessageType::MessageInfo, "Defined " + conv->toString());

    return conv;
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}