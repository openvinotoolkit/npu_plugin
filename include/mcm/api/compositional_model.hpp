/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_COMPOSITIONAL_MODEL_HPP_
#define MV_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/op_model.hpp"

namespace mv

{

    class CompositionalModel : private OpModel
    {

    public:

        CompositionalModel(OpModel& model);
        virtual ~CompositionalModel();

        virtual Data::TensorIterator add(Data::TensorIterator data0, Data::TensorIterator data1, const std::string&name = "");
        virtual Data::TensorIterator averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name = "");
        virtual Data::TensorIterator batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const std::string&name = "");
        virtual Data::TensorIterator bias(Data::TensorIterator data, Data::TensorIterator weights, const std::string&name = "");
        virtual Data::TensorIterator constant(const DType& dType, const std::vector<double>& data, const Order& order, const Shape& shape, const std::string&name = "");
        virtual Data::TensorIterator conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name = "");
        virtual Data::TensorIterator input(const DType& dType, const Order& order, const Shape& shape, const std::string&name = "");
        virtual Data::TensorIterator output(Data::TensorIterator data, const std::string&name = "");

        using OpModel::getSourceOp;
        using OpModel::addAttr;
        using OpModel::isValid;

    };

}

#endif //MV_COMPOSITIONAL_MODEL_HPP_
