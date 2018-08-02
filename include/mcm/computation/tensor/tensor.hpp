#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <functional>
#include <exception>
#include <memory>
#include <algorithm>
#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{

    class ShapeError : public std::logic_error
    {

        public:
            explicit ShapeError(const std::string& whatArg);
        
    };

    class OrderError : public std::logic_error
    {

        public:
            explicit OrderError(const std::string& whatArg);
        
    };

    class ValueError : public std::logic_error
    {

        public:
            explicit ValueError(const std::string& whatArg);

    };
    
    class Tensor : public ComputationElement
    {

        static allocator allocator_;
        std::shared_ptr<dynamic_vector<float_type>> data_;
        float_type errValue;
        Shape shape_;
        bool populated_;
        static static_vector<dim_type, byte_type, max_ndims> subsBuffer_;
        std::function<unsigned(const Shape&, const static_vector<dim_type, byte_type, max_ndims>&)> subToIndFcn_;
        std::function<static_vector<dim_type, byte_type, max_ndims>(const Shape& s, unsigned)> indToSubFcn_;

        static const std::function<unsigned(const Shape& s, const static_vector<dim_type, byte_type, max_ndims>&)> subToIndColumMajor_;
        static const std::function<static_vector<dim_type, byte_type, max_ndims>(const Shape& s, unsigned)> indToSubColumMajor_;
        static const std::function<unsigned(const Shape& s, const static_vector<dim_type, byte_type, max_ndims>&)> subToIndRowMajor_;
        static const std::function<static_vector<dim_type, byte_type, max_ndims>(const Shape& s, unsigned)> indToSubRowMajor_;
        static const std::function<unsigned(const Shape& s, const static_vector<dim_type, byte_type, max_ndims>&)> subToIndPlanar_;
        static const std::function<static_vector<dim_type, byte_type, max_ndims>(const Shape& s, unsigned)> indToSubPlanar_;

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

        static inline std::function<unsigned(const Shape&, const static_vector<dim_type, byte_type, max_ndims>&)> selectSubToInd_(Order order)
        {
            switch (order)
            {
                case Order::ColumnMajor:
                    return subToIndColumMajor_;

                case Order::RowMajor:
                    return subToIndRowMajor_;

                case Order::Planar:
                    return subToIndPlanar_;

                default:
                    throw OrderError("Undefined order");
            }
        }

        static inline std::function<static_vector<dim_type, byte_type, max_ndims>(const Shape& s, unsigned)> selectIndToSub_(Order order)
        {
            switch (order)
            {
                case Order::ColumnMajor:
                    return indToSubColumMajor_;

                case Order::RowMajor:
                    return indToSubRowMajor_;

                case Order::Planar:
                    return indToSubPlanar_;

                default:
                    throw OrderError("Undefined order");
            }
        }

        inline unsigned subToInd_(const Shape& s, static_vector<dim_type, byte_type, max_ndims>& sub) const
        {
            return subToIndFcn_(s, sub);
        }

        inline static_vector<dim_type, byte_type, max_ndims> indToSub_(const Shape& s, unsigned idx) const
        {
            return indToSubFcn_(s, idx);
        }

    public:

        static unsigned subToInd(const Shape& shape, const static_vector<dim_type, byte_type, max_ndims>& sub, Order order);
        static static_vector<dim_type, byte_type, max_ndims> indToSub(const Shape& s, unsigned idx, Order order);

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

        inline unsigned subToInd(const static_vector<dim_type, byte_type, max_ndims>& sub) const
        {
            return subToIndFcn_(shape_, sub);
        }

        static_vector<dim_type, byte_type, max_ndims> indToSub(unsigned idx) const
        {
            return indToSubFcn_(shape_, idx);
        }

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

            return (*data_)[subToInd(subsBuffer_)];

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

            return (*data_)[subToInd(subsBuffer_)];

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
