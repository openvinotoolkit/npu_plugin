#ifndef MATH_HPP_
#define MATH_HPP_

#include "include/mcm/computation/tensor/tensor.hpp"
#include <cmath>

namespace mv
{

    namespace math
    {
       
        Tensor add(const Tensor& t1, const Tensor& t2);
        Tensor add(const Tensor& t, float_type x);
        Tensor subtract(const Tensor& t1, const Tensor& t2);
        Tensor subtract(const Tensor& t, float_type x);
        Tensor multiply(const Tensor& t1, const Tensor& t2);
        Tensor multiply(const Tensor& t, float_type x);
        Tensor divide(const Tensor& t1, const Tensor& t2);
        Tensor divide(const Tensor& t, float_type x);
        Tensor sqrt(const Tensor& t);

    }

}

#endif // MATH_HPP_