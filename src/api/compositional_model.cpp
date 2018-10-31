/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "include/mcm/api/compositional_model.hpp"

mv::CompositionalModel::CompositionalModel(OpModel& model) :
    OpModel(model)
{

}

mv::CompositionalModel::~CompositionalModel()
{

}

mv::Data::TensorIterator mv::CompositionalModel::add(Data::TensorIterator data0, Data::TensorIterator data1, const std::string&name)
{
    return defineOp(
        "Add",
        {
            data0,
            data1
        },
        {
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name)
{
    return defineOp(
        "AveragePool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "padding", padding },
            { "stride", stride }
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const std::string&name)
{
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
            { "eps", eps }
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::bias(Data::TensorIterator data, Data::TensorIterator weights, const std::string&name)
{
    return defineOp(
        "Bias",
        {
            data,
            weights
        },
        {
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::constant(const DType& dType, const std::vector<double>& data, const Order& order, const Shape& shape, const std::string&name)
{
    return defineOp(
        "Constant",
        {
        },
        {
            { "dType", dType },
            { "data", data },
            { "order", order },
            { "shape", shape }
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name)
{
    return defineOp(
        "Conv",
        {
            data,
            weights
        },
        {
            { "padding", padding },
            { "stride", stride }
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::input(const DType& dType, const Order& order, const Shape& shape, const std::string&name)
{
    return defineOp(
        "Input",
        {
        },
        {
            { "dType", dType },
            { "order", order },
            { "shape", shape }
        },
        name
    );
}

mv::Data::TensorIterator mv::CompositionalModel::output(Data::TensorIterator data, const std::string&name)
{
    return defineOp(
        "Output",
        {
            data
        },
        {
        },
        name
    );
}

