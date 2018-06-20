#ifndef MATH_HPP_
#define MATH_HPP_

#include "include/mcm/computation/tensor/tensor.hpp"
#include <cmath>

namespace mv
{

    namespace math
    {

        auto elementAdd = [](float_type x, float_type y) { return x + y; };
        auto elementSubtract = [](float_type x, float_type y) { return x - y; };
        auto elementMulitply = [](float_type x, float_type y) { return x * y; };
        auto elementDivide = [](float_type x, float_type y) { return x / y; };

       
        Tensor add(const Tensor& t1, const Tensor& t2);
        Tensor add(const Tensor& t, float_type x);
        Tensor subtract(const Tensor& t1, const Tensor& t2);
        Tensor subtract(const Tensor& t, float_type x);
        Tensor multiply(const Tensor& t1, const Tensor& t2);
        Tensor multiply(const Tensor& t, float_type x);
        Tensor divide(const Tensor& t1, const Tensor& t2);
        Tensor divide(const Tensor& t, float_type x);
        Tensor sqrt(const Tensor& t);

        template <typename Functor>
        Shape elementWise(const Tensor& t1, const Tensor& t2, Functor& opFunc, dynamic_vector<float_type> &outputData)
        {

            if (!t1.isPopulated() || !t2.isPopulated())
            {
                Tensor::logger().log(Logger::MessageType::MessageError, "Unable to perfom element-wise operation using unpopulated tensor");
                return Shape();
            }

            Shape s1 = t1.getShape(), s2 = t2.getShape();

            if (s1 == s2)
            {
                if (outputData.size() != s1.totalSize())
                {
                    outputData.resize(s1.totalSize());
                }
                
                for (unsigned i = 0; i < outputData.size(); ++i)
                    outputData[i] = opFunc(t1(i), t2(i));
                return s1;
            }
            else
            {   

                Shape sO = Shape::broadcast(s1, s2);

                if (sO.ndims() == 0)
                    return Shape();

                if (s1.ndims() > s2.ndims())
                {
                    s2 = Shape::augment(s2, s1.ndims());
                }
                else if (s2.ndims() > s1.ndims())
                    s1 = Shape::augment(s1, s2.ndims());

                
                std::vector<bool> b1(sO.ndims()), b2(sO.ndims());

                for (unsigned i = 0; i < sO.ndims(); ++i)
                {

                    if (s1[i] == 1 && sO[i] > 1)
                        b1[i] = true;
                    else
                        b1[i] = false;
                    
                    if (s2[i] == 1 && sO[i] > 1)
                        b2[i] = true;
                    else
                        b2[i] = false;

                }

                if (outputData.size() != sO.totalSize())
                {
                    outputData.resize(sO.totalSize());
                }

                for (unsigned i = 0; i < outputData.size(); ++i)
                {
                    
                    dynamic_vector<unsigned> subO = Tensor::indToSub(sO, i);
                    dynamic_vector<unsigned> sub1 = subO, sub2 = subO;

                    for (unsigned j = 0; j < subO.size(); ++j)
                    {
                        if (b1[j])
                            sub1[j] = 0;
                        if (b2[j])
                            sub2[j] = 0;
                    }

                    outputData[i] = opFunc(t1(Tensor::subToInd(s1, sub1)), t2(Tensor::subToInd(s2, sub2)));

                }

                return sO; 

            }

            return Shape();

        }

    }

}

#endif // MATH_HPP_