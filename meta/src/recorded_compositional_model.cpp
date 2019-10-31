/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "meta/include/mcm/recorded_compositional_model.hpp"

mv::RecordedCompositionalModel::RecordedCompositionalModel(CompositionalModel& model, const std::string& outputPath, const std::string tab) :
    model_(model),
    srcStream_(outputPath + model_.getName() + ".cpp", std::ios::out | std::ios::trunc),
    tab_(tab)
{
    srcStream_ << "#include \"/home/tbartsok/Desktop/WORK/mcmCompiler" << "/meta/include/mcm/compositional_model.hpp\""<< std::endl;
    srcStream_ << "#include \"/home/tbartsok/Desktop/WORK/mcmCompiler" << "/include/mcm/computation/model/op_model.hpp\""<< std::endl;
    srcStream_ << std::endl << "int main()" << std::endl << "{" << std::endl << tab_ << "using namespace mv;" << std::endl;
    srcStream_ << tab_ << "OpModel model(\"" << getName() << "\");" << std::endl;
}

mv::RecordedCompositionalModel::~RecordedCompositionalModel()
{
    srcStream_ << std::endl << tab_ << "return 0;" << std::endl << "}" << std::endl;
    srcStream_.close();
}

mv::Data::TensorIterator mv::RecordedCompositionalModel:: add(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.add(inputs, quantParams, name);
    std::string input0 = inputs[0]->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.add(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.averagePool(data, kSize, stride, padding, exclude_pad, auto_pad, rounding_type, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.averagePool(" << input0 << ", " << Attribute(kSize).toLongString() << ", " << Attribute(stride).toLongString() << ", " << Attribute(padding).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.batchNormalization(data, mean, variance, offset, scale, eps, quantParams, name);
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
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.batchNormalization(" << input0<< ", " << input1<< ", " << input2<< ", " << input3<< ", " << input4 << ", " << Attribute(eps).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: bias(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.bias(data, weights, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.bias(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: clamp(Data::TensorIterator data, const double& min, const double& max, const std::string& name)
{
    Data::TensorIterator output = model_.clamp(data, min, max, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.clamp(" << input0 << ", " << Attribute(min).toLongString() << ", " << Attribute(max).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.concat(inputs, axis, quantParams, name);
    std::string input0 = inputs[0]->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.concat(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.constant(data, shape, dType, order, quantParams, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.constant(" << Attribute(data).toLongString() << ", " << Attribute(shape).toLongString() << ", " << Attribute(dType).toLongString() << ", " << Attribute(order).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.constantInt(data, shape, dType, order, quantParams, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.constantInt(" << Attribute(data).toLongString() << ", " << Attribute(shape).toLongString() << ", " << Attribute(dType).toLongString() << ", " << Attribute(order).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.conv(data, weights, stride, padding, dilationFactor, group, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.conv(" << input0<< ", " << input1 << ", " << Attribute(stride).toLongString() << ", " << Attribute(padding).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: copy(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.copy(data, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.copy(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.depthwiseConv(data, weights, stride, padding, dilationFactor, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.depthwiseConv(" << input0<< ", " << input1 << ", " << Attribute(stride).toLongString() << ", " << Attribute(padding).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: divide(Data::TensorIterator data0, Data::TensorIterator data1, const std::string& name)
{
    Data::TensorIterator output = model_.divide(data0, data1, name);
    std::string input0 = data0->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = data1->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.divide(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: dropout(Data::TensorIterator input, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.dropout(input, quantParams, name);
    std::string input0 = input->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.dropout(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: elu(Data::TensorIterator data, const unsigned& alpha, const std::string& name)
{
    Data::TensorIterator output = model_.elu(data, alpha, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.elu(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.fullyConnected(data, weights, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.fullyConnected(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: identity(Data::TensorIterator data, const std::string& name)
{
    Data::TensorIterator output = model_.identity(data, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.identity(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.input(shape, dType, order, quantParams, name);
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.input(" << Attribute(shape).toLongString() << ", " << Attribute(dType).toLongString() << ", " << Attribute(order).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: leakyRelu(Data::TensorIterator data, const double& alpha, const std::string& name)
{
    Data::TensorIterator output = model_.leakyRelu(data, alpha, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.leakyRelu(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name)
{
    Data::TensorIterator output = model_.localResponseNormalization(data, size, bias, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.localResponseNormalization(" << input0 << ", " << Attribute(size).toLongString() << ", " << Attribute(bias).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: matMul(Data::TensorIterator data0, Data::TensorIterator data1, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.matMul(data0, data1, quantParams, name);
    std::string input0 = data0->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = data1->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.matMul(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.maxPool(data, kSize, stride, padding, exclude_pad, auto_pad, rounding_type, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.maxPool(" << input0 << ", " << Attribute(kSize).toLongString() << ", " << Attribute(stride).toLongString() << ", " << Attribute(padding).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: multiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.multiply(inputs, quantParams, name);
    std::string input0 = inputs[0]->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.multiply(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: output(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.output(data, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    srcStream_ << tab_ << "model.output(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: permute(Data::TensorIterator data, const Order& order, const std::string& name)
{
    Data::TensorIterator output = model_.permute(data, order, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.permute(" << input0 << ", " << Attribute(order).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name)
{
    Data::TensorIterator output = model_.prelu(data, slope, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = slope->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.prelu(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const std::string& name)
{
    Data::TensorIterator output = model_.regionYolo(data, coords, classes, do_softmax, num, mask, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.regionYolo(" << input0 << ", " << Attribute(coords).toLongString() << ", " << Attribute(classes).toLongString() << ", " << Attribute(do_softmax).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: relu(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.relu(data, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.relu(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: reorder(Data::TensorIterator data, const Order& order, const std::string& name)
{
    Data::TensorIterator output = model_.reorder(data, order, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.reorder(" << input0 << ", " << Attribute(order).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: reorgYolo(Data::TensorIterator data, const unsigned& stride, const std::string& name)
{
    Data::TensorIterator output = model_.reorgYolo(data, stride, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.reorgYolo(" << input0 << ", " << Attribute(stride).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: reshape(Data::TensorIterator data0, const Shape& shape, const std::string& order, const std::string& name)
{
    Data::TensorIterator output = model_.reshape(data0, shape, order, name);
    std::string input0 = data0->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.reshape(" << input0 << ", " << Attribute(shape).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: scale(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.scale(data, weights, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string input1 = weights->getName();
    std::transform(input1.begin(),input1.end(), input1.begin(), ::tolower);
    std::replace(input1.begin(), input1.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.scale(" << input0<< ", " << input1 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: sigmoid(Data::TensorIterator data, const std::string& name)
{
    Data::TensorIterator output = model_.sigmoid(data, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.sigmoid(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.slice(data, begin, size, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.slice(" << input0 << ", " << Attribute(begin).toLongString() << ", " << Attribute(size).toLongString() << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: softmax(Data::TensorIterator data, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.softmax(data, axis, quantParams, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.softmax(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: subtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    Data::TensorIterator output = model_.subtract(inputs, quantParams, name);
    std::string input0 = inputs[0]->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.subtract(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::TensorIterator mv::RecordedCompositionalModel:: tanh(Data::TensorIterator data, const std::string& name)
{
    Data::TensorIterator output = model_.tanh(data, name);
    std::string input0 = data->getName();
    std::transform(input0.begin(),input0.end(), input0.begin(), ::tolower);
    std::replace(input0.begin(), input0.end(), ':', '_');
    std::string output0 = output->getName();
    std::transform(output0.begin(),output0.end(), output0.begin(), ::tolower);
    std::replace(output0.begin(), output0.end(), ':', '_');
    srcStream_ << tab_ << "Data::TensorIterator " + output0 + " = " << "model.tanh(" << input0 << ");" << std::endl;
    return output;
}
mv::Data::OpListIterator mv::RecordedCompositionalModel::getSourceOp(Data::TensorIterator tensor)
{
    return model_.getSourceOp(tensor);
}
void mv::RecordedCompositionalModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    return model_.addAttr(op, name, attr);
}
bool mv::RecordedCompositionalModel::isValid() const
{
    return model_.isValid();
}
bool mv::RecordedCompositionalModel::isValid(Data::TensorIterator tensor) const
{
    return model_.isValid(tensor);
}
bool mv::RecordedCompositionalModel::isValid(Data::OpListIterator op) const
{
    return model_.isValid(op);
}
std::string mv::RecordedCompositionalModel::getName() const
{
    return model_.getName();
}
