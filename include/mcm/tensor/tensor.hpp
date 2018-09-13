#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <memory>
#include <algorithm>
#include <vector>
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/order.hpp"
#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/value_error.hpp"

namespace mv
{

    class Tensor : public Element
    {

        std::shared_ptr<std::vector<double>> data_;
        static std::vector<std::size_t> subsBuffer_;

        void elementWise_(const Tensor& other, const std::function<double(double, double)>& opFunc);

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
        ~Tensor();

        bool populate(const std::vector<double>& data);
        bool populate(const std::vector<double>& data, Order order);
        bool unpopulate();
        
        bool broadcast(const Shape& shape);

        std::vector<double>& getData();
        void setDType(DType dType);
        DType getDType() const;
        void setOrder(Order order);
        Order getOrder() const;
        
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
        double& at(unsigned idx);
        const double& at(unsigned idx) const;
        double& operator()(unsigned idx);
        const double& operator()(unsigned idx) const;
        double& operator()(const std::vector<std::size_t>& sub);
        const double& operator()(const std::vector<std::size_t>& sub) const;

        inline bool isPopulated() const
        {
            return get<bool>("populated");
        }

        inline Shape getShape() const
        {
            return get<Shape>("shape");
        }

        inline std::vector<std::size_t> indToSub(unsigned index)
        {
            return indToSub_(getShape(), index);
        }

        inline unsigned subToInd(const std::vector<std::size_t>& sub) const
        {
            return subToInd_(getShape(), sub);
        }

        inline std::vector<std::size_t> indToSub_(const Shape& s, unsigned index)
        {
            return getOrder().indToSub(s, index);
        }

        inline unsigned subToInd_(const Shape& s, const std::vector<std::size_t>& sub) const
        {
            return getOrder().subToInd(s, sub);
        }



        template<typename... Idx>
        double& at(Idx... indices)
        {

            return const_cast<double&>(static_cast<const Tensor*>(this)->at(indices...));

        }

        template<typename... Idx>
        const double& at(Idx... indices) const
        {

            if (!isPopulated())
                throw ValueError(*this, "Unable to access the data value for an unpopulated tensor");

            subsBuffer_.clear();
            unfoldSubs_(indices...);

            auto idx = getOrder().subToInd(getShape(), subsBuffer_);
            if (idx >= data_->size())
                throw IndexError(*this, idx, "Exceeds dimension of the data vector");
            return (*data_)[idx];

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

        std::string toString() const override;
        virtual std::string getLogID() const override;

    };

}

#endif // MODEL_TENSOR_HPP_
