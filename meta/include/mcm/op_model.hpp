/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_OP_MODEL_HPP_
#define MV_OP_MODEL_HPP_

#include "meta/include/mcm/compositional_model.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"

#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv

{

    class OpModel: public BaseOpModel, public CompositionalModel
    {

    public:

        OpModel(const std::string& name);
        OpModel(ComputationModel& model);
        virtual ~OpModel();

        mv::Data::TensorIterator add(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const std::string& auto_pad = "", const std::string& rounding_type = "floor", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator barrierTask(const Barrier& Barrier, const std::string& name = "");
        mv::Data::TensorIterator batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator bias(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator clamp(Data::TensorIterator data, const double& min, const double& max, const std::string& name = "") override;
        mv::Data::TensorIterator concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator constantDataElement(const std::vector<mv::DataElement>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
        mv::Data::TensorIterator constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator conversion(Data::TensorIterator data, const Order& order, const std::string& name = "");
        mv::Data::TensorIterator copy(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator dMATask(Data::TensorIterator data, const DmaDirection& direction, const std::string& name = "");
        mv::Data::TensorIterator dPUTaskConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
mv::Data::TensorIterator dPUTaskMaxPool(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const std::string& auto_pad = "", const std::string& rounding_type = "floor", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
mv::Data::TensorIterator dPUTaskDepthwiseConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
mv::Data::TensorIterator dPUTaskAdd(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
mv::Data::TensorIterator dPUTaskSubtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
mv::Data::TensorIterator dPUTaskMultiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
        mv::Data::TensorIterator deallocate(Data::TensorIterator inputs, const std::string& name = "");
        mv::Data::TensorIterator depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator divide(Data::TensorIterator data0, Data::TensorIterator data1, const std::string& name = "") override;
        mv::Data::TensorIterator dropout(Data::TensorIterator input, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator elu(Data::TensorIterator data, const unsigned& alpha = 1, const std::string& name = "") override;
        mv::Data::TensorIterator fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator identity(Data::TensorIterator data, const std::string& name = "") override;
        mv::Data::TensorIterator implicitConcat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
        mv::Data::TensorIterator input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator leakyRelu(Data::TensorIterator data, const double& alpha = 0, const std::string& name = "") override;
        mv::Data::TensorIterator localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name = "") override;
        mv::Data::TensorIterator matMul(Data::TensorIterator data0, Data::TensorIterator data1, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const std::string& auto_pad = "", const std::string& rounding_type = "floor", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator multiply(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator output(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator permute(Data::TensorIterator data, const Order& order, const std::string& name = "") override;
        mv::Data::TensorIterator placeholderTask(const Shape& shape, const DType& dType, const Order& order, const std::string& name = "");
        mv::Data::TensorIterator prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name = "") override;
        mv::Data::TensorIterator regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {}, const std::string& name = "") override;
        mv::Data::TensorIterator relu(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator reorder(Data::TensorIterator data, const Order& order, const std::string& name = "") override;
        mv::Data::TensorIterator reorgYolo(Data::TensorIterator data, const unsigned& stride, const std::string& name = "") override;
        mv::Data::TensorIterator reshape(Data::TensorIterator data0, const Shape& shape, const std::string& order = "", const std::string& name = "") override;
        mv::Data::TensorIterator scale(Data::TensorIterator data, Data::TensorIterator weights, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator sigmoid(Data::TensorIterator data, const std::string& name = "") override;
        mv::Data::TensorIterator slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator softmax(Data::TensorIterator data, const std::string& axis = "C", const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator sparsityMap(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "");
        mv::Data::TensorIterator subtract(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") override;
        mv::Data::TensorIterator tanh(Data::TensorIterator data, const std::string& name = "") override;
        mv::Data::TensorIterator weightsTable(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{},{},{}}, const std::string& name = "");

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;
        bool isValid() const override;
        bool isValid(Data::TensorIterator tensor) const override;
        bool isValid(Data::OpListIterator op) const override;
        std::string getName() const override;

    };

}

#endif //MV_OP_MODEL_HPP_
