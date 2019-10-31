/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "meta/include/mcm/op_model.hpp"

mv::OpModel::OpModel(const std::string& name) :
BaseOpModel(name)
{

}

mv::OpModel::OpModel(ComputationModel& other) :
BaseOpModel(other)
{

}

mv::OpModel::~OpModel()
{

}

mv::Data::TensorIterator mv::OpModel::add(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Add",
        inputs,
        {
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "AveragePool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::barrierTask(const Barrier& Barrier, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "BarrierTask",
        {
        },
        {
            { "Barrier", Barrier }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "BatchNormalization",
        {
            data,
            mean,
            variance,
            offset,
            scale
        },
        {
            { "eps", eps },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::bias(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Bias",
        {
            data,
            weights
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::clamp(Data::TensorIterator data, const double& min, const double& max, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Clamp",
        {
            data
        },
        {
            { "min", min },
            { "max", max }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Concat",
        inputs,
        {
            { "axis", axis },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Constant",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::constantDataElement(const std::vector<mv::DataElement>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ConstantDataElement",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ConstantInt",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Conv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::conversion(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Conversion",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::copy(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Copy",
        {
            data
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dMATask(Data::TensorIterator data, const DmaDirection& direction, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DMATask",
        {
            data
        },
        {
            { "direction", direction }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Conv") },
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskMaxPool(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("MaxPool") },
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskDepthwiseConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("DepthwiseConv") },
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskAdd(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Add") },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskSubtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Subtract") },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskMultiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Multiply") },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::deallocate(Data::TensorIterator inputs, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Deallocate",
        {
            inputs
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DepthwiseConv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::divide(Data::TensorIterator data0, Data::TensorIterator data1, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Divide",
        {
            data0,
            data1
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dropout(Data::TensorIterator input, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Dropout",
        {
            input
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::elu(Data::TensorIterator data, const unsigned& alpha, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Elu",
        {
            data
        },
        {
            { "alpha", alpha }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "FullyConnected",
        {
            data,
            weights
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::identity(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Identity",
        {
            data
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::implicitConcat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ImplicitConcat",
        inputs,
        {
            { "axis", axis },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Input",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::leakyRelu(Data::TensorIterator data, const double& alpha, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "LeakyRelu",
        {
            data
        },
        {
            { "alpha", alpha }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "LocalResponseNormalization",
        {
            data
        },
        {
            { "size", size },
            { "bias", bias }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::matMul(Data::TensorIterator data0, Data::TensorIterator data1, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "MatMul",
        {
            data0,
            data1
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "MaxPool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::multiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Multiply",
        inputs,
        {
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Output",
        {
            data
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::permute(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Permute",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::placeholderTask(const Shape& shape, const DType& dType, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "PlaceholderTask",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Prelu",
        {
            data,
            slope
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "RegionYolo",
        {
            data
        },
        {
            { "coords", coords },
            { "classes", classes },
            { "do_softmax", do_softmax },
            { "num", num },
            { "mask", mask }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Relu",
        {
            data
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reorder(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Reorder",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reorgYolo(Data::TensorIterator data, const unsigned& stride, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ReorgYolo",
        {
            data
        },
        {
            { "stride", stride }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator data0, const Shape& shape, const std::string& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Reshape",
        {
            data0
        },
        {
            { "shape", shape },
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Scale",
        {
            data,
            weights
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::sigmoid(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Sigmoid",
        {
            data
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Slice",
        {
            data
        },
        {
            { "begin", begin },
            { "size", size },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator data, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Softmax",
        {
            data
        },
        {
            { "axis", axis },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::sparsityMap(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "SparsityMap",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::subtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Subtract",
        inputs,
        {
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::tanh(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Tanh",
        {
            data
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::weightsTable(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "WeightsTable",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return BaseOpModel::getSourceOp(tensor);
}
void mv::OpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    return BaseOpModel::addAttr(op, name, attr);
}
bool mv::OpModel::isValid() const
{
    return BaseOpModel::isValid();
}
bool mv::OpModel::isValid(Data::TensorIterator tensor) const
{
    return BaseOpModel::isValid(tensor);
}
bool mv::OpModel::isValid(Data::OpListIterator op) const
{
    return BaseOpModel::isValid(op);
}
std::string mv::OpModel::getName() const
{
    return BaseOpModel::getName();
}
