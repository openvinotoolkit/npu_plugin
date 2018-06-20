#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/model/element.hpp"

namespace mv
{
    
    class Tensor : public ComputationElement
    {

        static allocator allocator_;
        allocator::owner_ptr<dynamic_vector<float_type>> data_;
        float_type errValue;

        static inline void unfoldIndex_(unsigned_type& currentResult, const Shape& shape, byte_type dim, unsigned_type idx)
        {
            assert(idx < shape[dim] && "Index exceeds dimension of tensor");
            currentResult = idx + shape[dim] * currentResult;
        }

        template<typename... Idx>
        static inline void unfoldIndex_(unsigned_type& currentResult, const Shape& shape, byte_type dim, unsigned_type idx, Idx... indices)
        {
            assert(idx < shape[dim] && "Index exceeds dimension of tensor");
            currentResult = idx + shape[dim] * currentResult;
            unfoldIndex_(currentResult, shape, dim + 1, indices...);
        }

    public:

        Tensor(const string &name, const Shape &shape, DType dType, Order order);
        Tensor(const string &name, const Shape &shape, DType dType, Order order, allocator::owner_ptr<dynamic_vector<float_type>> data);
        Tensor(const string &name, const Shape &shape, DType dType, Order order, dynamic_vector<float_type>& data);
        Tensor(const Tensor &other);
        Tensor();
        bool populate(dynamic_vector<float_type>& data);
        bool unpopulate();
        bool isPopulated() const;
        dynamic_vector<float_type> &getData();
        Shape getShape() const;
        DType getDType() const;
        Order getOrder() const;
        string toString() const;
        static Logger& logger();
        
        bool add(const Tensor& other);
        bool subtract(const Tensor& other);
        bool mulitply(const Tensor& other);
        bool divide(const Tensor& other);

        template<typename... Idx>
        static unsigned subToInd(const Shape& s, Idx... indices)
        {

            assert(sizeof...(Idx) == s.ndims() && "Number of indices does not match ndims of tensor");
            byte_type dim = 0u;
            unsigned_type result = 0u;
            unfoldIndex_(result, s, dim, indices...);
            return result;

        }

        static unsigned subToInd(const Shape& shape, const dynamic_vector<unsigned>& sub)
        {
            assert(sub.size() == shape.ndims() && "Shape and subs size mismatch");
            assert(sub.size() != 0 && "Cannot compute index for an empty tensor");

            unsigned currentResult = 0;

            for (unsigned i = 0; i < sub.size(); ++i)
            {
                assert(sub[i] < shape[i] && "Index exceeds dimension of tensor");
                currentResult = sub[i] + shape[i] * currentResult;
            }

            return currentResult;

        }

        static dynamic_vector<unsigned> indToSub(const Shape& s, unsigned idx)
        {

            dynamic_vector<unsigned> sub(s.ndims());
            sub[s.ndims() - 1] =  idx % s[s.ndims() - 1];
            int offset = -sub[s.ndims() - 1];
            int scale = s[s.ndims() - 1];
            for (int i = s.ndims() - 2; i >= 0; --i)
            {   
                sub[i] = (idx + offset) / scale % s[i];
                offset -= sub[i] * s[i + 1];
                scale *= s[i];
            }

            return sub;

        }

        template<typename... Idx>
        unsigned subToInd(Idx... indices) const
        {

            return subToInd(getShape(), indices...);

        }

        unsigned subToInd(const dynamic_vector<unsigned>& sub) const
        {
            return subToInd(getShape(), sub);
        }

        dynamic_vector<unsigned> indToSub(unsigned idx) const
        {

            return indToSub(getShape(), idx);

        }

        float_type& at(const dynamic_vector<unsigned>& sub)
        {
            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[subToInd(sub)];
        }

        float_type at(const dynamic_vector<unsigned>& sub) const
        {
            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[subToInd(sub)];
        }

        template<typename... Idx>
        float_type& at(Idx... indices)
        {
            
            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[subToInd(indices...)];

        }

        template<typename... Idx>
        float_type at(Idx... indices) const
        {
            
            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[subToInd(indices...)];

        }

        float_type& at(unsigned idx)
        {
            
            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[idx];

        }

        float_type at(unsigned idx) const
        {

            if (!isPopulated())
            {
                logger_.log(Logger::MessageType::MessageError, "Attempt of reading a value from an unpopulated tensor");
                return errValue;
            }

            return (*data_)[idx];

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

        float_type& operator()(unsigned idx)
        {
            return at(idx);
        }

        float_type operator()(unsigned idx) const
        {
            return at(idx);
        }
        
        float_type& operator()(const dynamic_vector<unsigned>& sub)
        {
            return at(sub);
        }

        float_type operator()(const dynamic_vector<unsigned>& sub) const
        {
            return at(sub);
        }
        
    };

}

#endif // MODEL_TENSOR_HPP_