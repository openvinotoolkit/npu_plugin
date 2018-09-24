#ifndef MATH_HPP_
#define MATH_HPP_

#include "include/mcm/tensor/tensor.hpp"
#include <cmath>

namespace mv
{

    namespace math
    {
       
        Tensor add(const Tensor& t1, const Tensor& t2);
        Tensor add(const Tensor& t, double x);
        Tensor subtract(const Tensor& t1, const Tensor& t2);
        Tensor subtract(const Tensor& t, double x);
        Tensor multiply(const Tensor& t1, const Tensor& t2);
        Tensor multiply(const Tensor& t, double x);
        Tensor divide(const Tensor& t1, const Tensor& t2);
        Tensor divide(const Tensor& t, double x);
        Tensor sqrt(const Tensor& t);

    }

}

#endif // MATH_HPP_