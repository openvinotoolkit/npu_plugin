/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_COMPOSITIONAL_MODEL_HPP_
#define MV_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"

#include "include/mcm/tensor/quantization_params.hpp"

namespace mv

{

    class CompositionalModel
    {

    public:

        virtual ~CompositionalModel() = 0;

        virtual mv::Data::TensorIterator add(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const std::string& auto_pad = "", const std::string& rounding_type = "floor", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator bias(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator clamp(Data::TensorIterator data, const double& min, const double& max, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator copy(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator divide(Data::TensorIterator data0, Data::TensorIterator data1, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator dropout(Data::TensorIterator input, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator elu(Data::TensorIterator data, const unsigned& alpha = 1, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator identity(Data::TensorIterator data, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator leakyRelu(Data::TensorIterator data, const double& alpha = 0, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator matMul(Data::TensorIterator data0, Data::TensorIterator data1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const std::string& auto_pad = "", const std::string& rounding_type = "floor", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator multiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator output(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator permute(Data::TensorIterator data, const Order& order, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator relu(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reorder(Data::TensorIterator data, const Order& order, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reorgYolo(Data::TensorIterator data, const unsigned& stride, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reshape(Data::TensorIterator data0, const Shape& shape, const std::string& order = "", const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator scale(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator sigmoid(Data::TensorIterator data, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator softmax(Data::TensorIterator data, const std::string& axis = "C", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator subtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator tanh(Data::TensorIterator data, const std::string& name = "") = 0;

        virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;
        virtual void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;
        virtual bool isValid(Data::TensorIterator tensor) const = 0;
        virtual bool isValid(Data::OpListIterator op) const = 0;
        virtual std::string getName() const = 0;

    };

}

#endif //MV_COMPOSITIONAL_MODEL_HPP_
