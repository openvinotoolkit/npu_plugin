/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateRecordedCompositionAPI()
*/

#ifndef MV_RECORDED_COMPOSITIONAL_MODEL_HPP_
#define MV_RECORDED_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/api/compositional_model.hpp"

namespace mv

{

    class RecordedCompositionalModel : public CompositionalModel
    {

    public:

        RecordedCompositionalModel(CompositionalModel& model);
        virtual ~RecordedCompositionalModel();

        Data::TensorIterator add(Data::TensorIterator data0, Data::TensorIterator data1, const std::string&name = "") override;
        Data::TensorIterator averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name = "") override;
        Data::TensorIterator batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const std::string&name = "") override;
        Data::TensorIterator bias(Data::TensorIterator data, Data::TensorIterator weights, const std::string&name = "") override;
        Data::TensorIterator constant(const DType& dType, const std::vector<double>& data, const Order& order, const Shape& shape, const std::string&name = "") override;
        Data::TensorIterator conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 4>& padding, const std::array<unsigned short, 2>& stride, const std::string&name = "") override;
        Data::TensorIterator input(const DType& dType, const Order& order, const Shape& shape, const std::string&name = "") override;
        Data::TensorIterator output(Data::TensorIterator data, const std::string&name = "") override;
    };

}

#endif //MV_RECORDED_COMPOSITIONAL_MODEL_HPP_
