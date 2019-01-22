#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <memory>
#include <algorithm>
#include <vector>
#include <iterator>
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/value_error.hpp"

namespace mv
{

    class Tensor : public Element
    {

        std::vector<double> data_;
        std::size_t blockSize_;
        std::vector<std::vector<double>::iterator> blocks_;
        Shape shape_;
        Order internalOrder_;

        void elementWise_(const Tensor& other, const std::function<double(double, double)>& opFunc);

        std::vector<std::size_t> indToSub_(const Shape& s, unsigned index) const;
        unsigned subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const;

    public:

        Tensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data);
        Tensor(const Tensor& other);
        ~Tensor();

        void populate(const std::vector<double>& data);
        void populate(const std::vector<double>& data, Order order);
        void unpopulate();

        /**
         * @brief Binds the data (values vector) of this tensor (slave) to the given master tensor. After this operation data accessed
         * from this tensor will be actually read/written to the master tensor. Using the leftPadding and rightPadding it is possible
         * to select a fragment of the master tensor. Shape of the calling tensor will be modified according to the shape of master tensor
         * and padding values. Data type and data order will be inherited from the master tensor. Automatically sets populated flag.
         * Current implementation will disallow any further reordering (setOrder()) and broadcasting (broadcast()) of both master and slave.
         * @param other Master tensor, must be populated
         * @param leftPadding Vector of values specifing the padding between the bounderies (left-top) of the master tensor and this tensor per dimenision.
         * @param rightPadding Vector of values specifing the padding between the bounderies (right-bottom) of the master tensor and this tensor per dimenision.
         */
        void bindData(Tensor& other, const std::vector<std::size_t>& leftPadding = {}, const std::vector<std::size_t>& rightPadding = {});
        void broadcast(const Shape& shape);

        std::vector<double> getData();
        void setDType(DType dType);
        DType getDType() const;
        void setOrder(Order order);
        Order getOrder() const;
        void setShape(const Shape& shape);
        
        void add(const Tensor& other);
        void add(double val);
        void subtract(const Tensor& other);
        void subtract(double val);
        void multiply(const Tensor& other);
        void multiply(double val);
        void divide(const Tensor& other);
        void divide(double val);
        void sqrt();

        double& at(const std::vector<std::size_t>& sub);
        const double& at(const std::vector<std::size_t>& sub) const;
        double& at(std::size_t idx);
        const double& at(std::size_t idx) const;
        double& operator()(std::size_t idx);
        const double& operator()(std::size_t idx) const;
        double& operator()(const std::vector<std::size_t>& sub);
        const double& operator()(const std::vector<std::size_t>& sub) const;

        inline bool isPopulated() const
        {
            return get<bool>("populated");
        }

        inline Shape& getShape()
        {
            return shape_;
        }

        inline const Shape& getShape() const
        {
            return shape_;
        }

        inline std::vector<std::size_t> indToSub(unsigned index) const
        {
            return indToSub_(getShape(), index);
        }

        inline unsigned subToInd(const std::vector<std::size_t>& sub) const
        {
            return subToInd_(getShape(), sub);
        }

        Tensor& operator=(const Tensor& other);
        std::vector<unsigned> computeStrides() const;

        std::string toString() const override;
        virtual std::string getLogID() const override;

        BinaryData toBinary();

    };

}

#endif // MODEL_TENSOR_HPP_
