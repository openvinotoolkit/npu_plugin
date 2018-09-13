#ifndef MV_COMPOSITIONAL_MODEL_HPP_
#define MV_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{

    class CompositionalModel
    {
        
    public:

        virtual ~CompositionalModel() = 0;

        virtual Data::TensorIterator input(const Shape& shape, DType dType, Order order, const std::string& name = "") = 0;

        virtual Data::TensorIterator output(Data::TensorIterator input, const std::string& name = "") = 0;

        virtual Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, DType dType,
            Order order, const std::string& name = "") = 0;

        virtual Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, 
            std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") = 0;

        virtual Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator maxpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize,
             std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") = 0;

        virtual Data::TensorIterator avgpool2D(Data::TensorIterator input, std::array<unsigned short, 2> kernelSize,
             std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string& name = "") = 0;

        virtual Data::TensorIterator concat(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;
            
        virtual Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, 
            Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, 
            double varianceEps, const std::string& name = "") = 0;

        virtual Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator relu(Data::TensorIterator input, const std::string& name = "") = 0;

        virtual Data::TensorIterator softmax(Data::TensorIterator input, const std::string& name = "") = 0;

        virtual Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, 
            const std::string& name = "") = 0;

        virtual Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, 
            const std::string& name = "") = 0;

        virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;

        virtual void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) = 0;
        
        virtual bool isValid() const = 0;
        virtual bool isValid(const Data::TensorIterator& it) const = 0;
        virtual bool isValid(const Data::OpListIterator& it) const = 0;
    };

}

#endif //MV_COMPOSITIONAL_MODEL_HPP_
