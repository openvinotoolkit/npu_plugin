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

const mv::OpListIterator mv::ComputationModel::input(const Shape &shape, Tensor::DType dType, Tensor::Order order, const string &name)
{

    string inputName = name;

    if (inputName.empty())
        inputName = "input";

    input_ = ops_graph.node_insert(ComputationOp(logger_, shape, shape, dType, order, inputName));

    logger_.log(Logger::MessageInfo, "Input '" + inputName + "' of size " + shape.toString() + " defined");

    return input_;

}

const mv::OpListIterator mv::ComputationModel::output(OpListIterator &predecessor, const string &name)
{

    string outputName = name;

    if (outputName.empty())
        outputName = "output";

    auto outputShape = predecessor->getOutputShape();
    auto outputDType = predecessor->getDType();
    auto outputOrder = predecessor->getOrder();
    VariableTensor inputTensor(logger_, name + "_output", outputShape, outputDType, outputOrder);
    output_ = ops_graph.node_insert(predecessor, ComputationOp(logger_, outputShape, outputShape, outputDType, outputOrder, outputName), inputTensor);

    logger_.log(Logger::MessageInfo, "Output '" + outputName + "' of size " + outputShape.toString() + " defined");

    return output_;

}

mv::OpListIterator mv::ComputationModel::convolutional(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, const string &name)
{

    string convName = name;

    if (convName.empty())
        convName = "conv";

    auto inputShape = predecessor->getOutputShape();
    auto convDType = predecessor->getDType();
    auto convOrder = predecessor->getOrder();

    Shape convShape(inputShape[0], inputShape[1] / strideX, inputShape[2] / strideY, weights.getShape()[2]);

    VariableTensor inputTensor(logger_, name + "_input", inputShape, convDType, convOrder);

    OpListIterator conv = ops_graph.node_insert(predecessor, ComputationOp(logger_, inputShape, convShape, convDType, convOrder, convName), inputTensor);

    conv->addAttr<ConstantModelTensor>("weigths", ConstantModelTensor(logger_, convName + "_weights", weights));
    conv->addAttr<byte_type>("strideX", strideX);
    conv->addAttr<byte_type>("strideY", strideY);

    string weights_shape = weights.getShape().toString();

    logger_.log(Logger::MessageInfo, "Conv layer '" + convName + "' defined: input '" + predecessor->getName() + "'; kernel " + weights_shape + 
    "; sX " + Printable::toString(strideX)  + "; sY " +  Printable::toString(strideY)) ;

    return conv;
}

const mv::Logger &mv::ComputationModel::logger() const
{

    return logger_;

}