#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <memory>
#include <algorithm>
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

        static allocator allocator_;
        std::shared_ptr<dynamic_vector<float_type>> data_;
        float_type errValue;
        Shape shape_;
        bool populated_;
        static static_vector<dim_type, byte_type, max_ndims> subsBuffer_;

        bool elementWise_(const Tensor& other, const std::function<float(float, float)>& opFunc);

        static inline void unfoldSubs_(unsigned_type sub)
        {
            subsBuffer_.push_back(sub);
            //output[dim] = sub;
        }

        template<typename... Subs>
        static inline void unfoldSubs_(unsigned_type sub, Subs... subs)
        {
            //output[dim] = sub;
            subsBuffer_.push_back(sub);
            unfoldSubs_(subs...);
        }


    public:

        Tensor(const string &name, const Shape &shape, DType dType, Order order);
        Tensor(const string &name, const Shape &shape, DType dType, Order order, const dynamic_vector<float_type>& data);
        Tensor(const Tensor &other);
        Tensor();
        Tensor(json::Value &v);
        ~Tensor();
        bool populate(const dynamic_vector<float_type>& data, Order order = Order::Unknown);
        bool unpopulate();
        void reorder(Order order);
        bool broadcast(const Shape& shape);
        dynamic_vector<float_type> &getData();
        DType getDType() const;
        Order getOrder() const;
        void setOrder(Order order);
        string toString() const;
        static Logger& logger();

        bool add(const Tensor& other);
        bool add(float val);
        bool subtract(const Tensor& other);
        bool subtract(float val);
        bool multiply(const Tensor& other);
        bool multiply(float val);
        bool divide(const Tensor& other);
        bool divide(float val);
        bool sqrt();

        float_type& at(const static_vector<dim_type, byte_type, max_ndims>& sub);
        const float_type& at(const static_vector<dim_type, byte_type, max_ndims>& sub) const;
        float_type& at(unsigned idx);
        const float_type& at(unsigned idx) const;
        float_type& operator()(unsigned idx);
        const float_type& operator()(unsigned idx) const;
        float_type& operator()(const static_vector<dim_type, byte_type, max_ndims>& sub);
        const float_type& operator()(const static_vector<dim_type, byte_type, max_ndims>& sub) const;

        inline bool isPopulated() const
        {
            return populated_;
        }

        inline Shape getShape() const
        {
            return shape_;
        }

        inline static_vector<dim_type, byte_type, max_ndims> indToSub(unsigned index)
        {
            return indToSub_(shape_, index);
        }

        inline unsigned subToInd(const static_vector<dim_type, byte_type, max_ndims>& sub) const
        {
            return subToInd_(shape_, sub);
        }

        inline static_vector<dim_type, byte_type, max_ndims> indToSub_(const Shape& s, unsigned index)
        {
            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return order_->indToSub(s, index);
        }

        inline unsigned subToInd_(const Shape& s, const static_vector<dim_type, byte_type, max_ndims>& sub) const
        {
            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return order_->subToInd(s, sub);
        }



        template<typename... Idx>
        float_type& at(Idx... indices)
        {

            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            //dynamic_vector<unsigned> subs(getShape().ndims());
            subsBuffer_.clear();
            unfoldSubs_(indices...);

            std::unique_ptr<OrderClass> order_ = mv::OrderFactory::createOrder(getOrder());

            return (*data_)[order_->subToInd(shape_, subsBuffer_)];

        }

        template<typename... Idx>
        float_type at(Idx... indices) const
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
        float_type& operator()(Idx... indices)
        {
            return at(indices...);
        }

        template<typename... Idx>
        float_type operator()(Idx... indices) const
        {
            return at(indices...);
        }

    };

}

#endif // MODEL_TENSOR_HPP_
