#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <memory>
#include <algorithm>
#include <vector>
#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/model/computation_element.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/base/exception/shape_error.hpp"
#include "include/mcm/base/exception/value_error.hpp"
#include "include/mcm/base/order/order_factory.hpp"

namespace mv
{

    class Tensor : public ComputationElement
    {

        std::shared_ptr<std::vector<double>> data_;
        double errValue;
        Shape shape_;
        bool populated_;
        static std::vector<std::size_t> subsBuffer_;

        bool elementWise_(const Tensor& other, const std::function<double(double, double)>& opFunc);

        static inline void unfoldSubs_(std::size_t sub)
        {
            subsBuffer_.push_back(sub);
            //output[dim] = sub;
        }

        template<typename... Subs>
        static inline void unfoldSubs_(std::size_t sub, Subs... subs)
        {
            //output[dim] = sub;
            subsBuffer_.push_back(sub);
            unfoldSubs_(subs...);
        }


    public:

        Tensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Tensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data);
        Tensor(const Tensor& other);
        Tensor();
        Tensor(json::Value& v);
        ~Tensor();
        bool populate(const std::vector<double>& data, Order order = Order::Unknown);
        bool unpopulate();
        void reorder(Order order);
        bool broadcast(const Shape& shape);
        std::vector<double>& getData();
        DType getDType() const;
        Order getOrder() const;
        void setOrder(Order order);
        std::string toString() const;
        static Logger& logger();

        bool add(const Tensor& other);
        bool add(double val);
        bool subtract(const Tensor& other);
        bool subtract(double val);
        bool multiply(const Tensor& other);
        bool multiply(double val);
        bool divide(const Tensor& other);
        bool divide(double val);
        bool sqrt();

        double& at(const std::vector<std::size_t>& sub);
        const double& at(const std::vector<std::size_t>& sub) const;
        double& at(unsigned idx);
        const double& at(unsigned idx) const;
        double& operator()(unsigned idx);
        const double& operator()(unsigned idx) const;
        double& operator()(const std::vector<std::size_t>& sub);
        const double& operator()(const std::vector<std::size_t>& sub) const;

        inline bool isPopulated() const
        {
            return populated_;
        }

        inline Shape getShape() const
        {
            return shape_;
        }

        inline std::vector<std::size_t> indToSub(unsigned index)
        {
            return indToSub_(shape_, index);
        }

        inline unsigned subToInd(const std::vector<std::size_t>& sub) const
        {
            return subToInd_(shape_, sub);
        }

        inline std::vector<std::size_t> indToSub_(const Shape& s, unsigned index)
        {
            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return order_->indToSub(s, index);
        }

        inline unsigned subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const
        {
            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return order_->subToInd(s, sub);
        }



        template<typename... Idx>
        double& at(Idx... indices)
        {

            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            //std::vector<unsigned> subs(getShape().ndims());
            subsBuffer_.clear();
            unfoldSubs_(indices...);

            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return (*data_)[order_->subToInd(shape_, subsBuffer_)];

        }

        template<typename... Idx>
        double at(Idx... indices) const
        {

            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            subsBuffer_.clear();
            unfoldSubs_(indices...);

            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return (*data_)[order_->subToInd(shape_, subsBuffer_)];

        }

        template<typename... Idx>
        double& operator()(Idx... indices)
        {
            return at(indices...);
        }

        template<typename... Idx>
        double operator()(Idx... indices) const
        {
            return at(indices...);
        }

    };

}

#endif // MODEL_TENSOR_HPP_
