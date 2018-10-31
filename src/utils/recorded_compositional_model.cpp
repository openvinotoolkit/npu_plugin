/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "include/mcm/utils/recorded_compositional_model.hpp"

mv::RecordedCompositionalModel::RecordedCompositionalModel(CompositionalModel& model) :
    CompositionalModel(model)
{

}

mv::RecordedCompositionalModel::~RecordedCompositionalModel()
{

}

mv::Data::TensorIterator mv::RecordedCompositionalModel::add(Data::TensorIterator data0, Data::TensorIterator data1, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::add(data0, data1, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::string input0 = data0->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = data1->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::cout << input0 << std::endl;
    std::cout << input1 << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::averagePool(data, kSize, padding, stride, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::cout << Attribute(kSize).toString() << std::endl;
    std::cout << Attribute(padding).toString() << std::endl;
    std::cout << Attribute(stride).toString() << std::endl;
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::cout << input0 << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::batchNormalization(data, mean, variance, offset, scale, eps, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::cout << Attribute(eps).toString() << std::endl;
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = mean->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string input2 = variance->getName();
    std::transform(input2.begin(),input2.end(), input2.begin(), ::tolower);
    std::replace(input2.begin(), input2.end(), ':', '_');
    std::string input3 = offset->getName();
    std::transform(input3.begin(),input3.end(), input3.begin(), ::tolower);
    std::replace(input3.begin(), input3.end(), ':', '_');
    std::string input4 = scale->getName();
    std::transform(input4.begin(),input4.end(), input4.begin(), ::tolower);
    std::replace(input4.begin(), input4.end(), ':', '_');
    std::cout << input0 << std::endl;
    std::cout << input1 << std::endl;
    std::cout << input2 << std::endl;
    std::cout << input3 << std::endl;
    std::cout << input4 << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::bias(Data::TensorIterator data, Data::TensorIterator weights, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::bias(data, weights, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::cout << input0 << std::endl;
    std::cout << input1 << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::constant(const DType& dType, const std::vector<double>& data, const Order& order, const Shape& shape, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::constant(dType, data, order, shape, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::cout << Attribute(dType).toString() << std::endl;
    std::cout << Attribute(data).toString() << std::endl;
    std::cout << Attribute(order).toString() << std::endl;
    std::cout << Attribute(shape).toString() << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::conv(data, weights, padding, stride, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::cout << Attribute(padding).toString() << std::endl;
    std::cout << Attribute(stride).toString() << std::endl;
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::cout << input0 << std::endl;
    std::cout << input1 << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::input(const DType& dType, const Order& order, const Shape& shape, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::input(dType, order, shape, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    std::cout << output0 << std::endl;
    std::cout << Attribute(dType).toString() << std::endl;
    std::cout << Attribute(order).toString() << std::endl;
    std::cout << Attribute(shape).toString() << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel::output(Data::TensorIterator data, const std::string&name)
{
    Data::TensorIterator output = CompositionalModel::output(data, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::cout << input0 << std::endl;
    return output;
}
