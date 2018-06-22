#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "include/mcm/computation/tensor/shape.hpp"
#include "include/mcm/computation/model/computation_element.hpp"

namespace mv
{
    
    class Tensor : public ComputationElement
    {

        static allocator allocator_;
        allocator::owner_ptr<dynamic_vector<float_type>> data_;
        float_type errValue;

        static inline void unfoldSubs_(dynamic_vector<unsigned> &output, byte_type &dim, unsigned_type sub)
        {
            output[dim] = sub;
        }

        template<typename... Subs>
        static inline void unfoldSubs_(dynamic_vector<unsigned> &output, byte_type &dim, unsigned_type sub, Subs... subs)
        {
            output[dim] = sub;
            unfoldSubs_(output, ++dim, subs...);
        }

    public:

        static unsigned subToInd(const Shape& shape, const dynamic_vector<unsigned>& sub);
        static dynamic_vector<unsigned> indToSub(const Shape& s, unsigned idx);

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

        unsigned subToInd(const dynamic_vector<unsigned>& sub) const;
        dynamic_vector<unsigned> indToSub(unsigned idx) const;
        float_type& at(const dynamic_vector<unsigned>& sub);
        float_type at(const dynamic_vector<unsigned>& sub) const;
        float_type& at(unsigned idx);
        float_type at(unsigned idx) const;
        float_type& operator()(unsigned idx);
        float_type operator()(unsigned idx) const;
        float_type& operator()(const dynamic_vector<unsigned>& sub);
        float_type operator()(const dynamic_vector<unsigned>& sub) const;
        
        template<typename... Idx>
        unsigned subToInd(Idx... indices) const
        {

            //return subToInd(getShape(), indices...);
            dynamic_vector<unsigned> subs(getShape().ndims());
            byte_type dim = 0;
            unfoldSubs_(subs, dim, indices...);
            return subToInd(getShape(), subs);

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