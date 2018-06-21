#include "include/mcm/computation/tensor/math.hpp"

mv::Tensor mv::math::add(const Tensor& t1, const Tensor& t2)
{

    dynamic_vector<float_type> data;
    auto outShape = elementWise(t1, t2, elementAdd, data);
    return Tensor("add_" + t1.getName() + "_" + t2.getName(), outShape, t1.getDType(), t1.getOrder(), data);

}

mv::Tensor mv::math::add(const Tensor& t, float_type x)
{
    dynamic_vector<float_type> data(t.getShape().totalSize());
    for (unsigned i = 0; i < t.getShape().totalSize(); ++i)
        data[i] = t(i) + x;
    return Tensor("add_" + t.getName() + "_scalar", t.getShape(), t.getDType(), t.getOrder(), data);
}

mv::Tensor mv::math::subtract(const Tensor& t1, const Tensor& t2)
{

    dynamic_vector<float_type> data;
    auto outShape =  elementWise(t1, t2, elementSubtract, data);
    return Tensor("sub_" + t1.getName() + "_" + t2.getName(), outShape, t1.getDType(), t1.getOrder(), data);

}

mv::Tensor mv::math::subtract(const Tensor& t, float_type x)
{
    dynamic_vector<float_type> data(t.getShape().totalSize());
    for (unsigned i = 0; i < t.getShape().totalSize(); ++i)
        data[i] = t(i) - x;
    return Tensor("sub_" + t.getName() + "_scalar", t.getShape(), t.getDType(), t.getOrder(), data);
}

mv::Tensor mv::math::multiply(const Tensor& t1, const Tensor& t2)
{

    dynamic_vector<float_type> data;
    auto outShape =  elementWise(t1, t2, elementMulitply, data);
    return Tensor("mul_" + t1.getName() + "_" + t2.getName(), outShape, t1.getDType(), t1.getOrder(), data);

}

mv::Tensor mv::math::multiply(const Tensor& t, float_type x)
{
    dynamic_vector<float_type> data(t.getShape().totalSize());
    for (unsigned i = 0; i < t.getShape().totalSize(); ++i)
        data[i] = t(i) * x;
    return Tensor("mul_" + t.getName() + "_scalar", t.getShape(), t.getDType(), t.getOrder(), data);
}

mv::Tensor mv::math::divide(const Tensor& t1, const Tensor& t2)
{

    dynamic_vector<float_type> data;
    auto outShape =  elementWise(t1, t2, elementDivide, data);
    return Tensor("div_" + t1.getName() + "_" + t2.getName(), outShape, t1.getDType(), t1.getOrder(), data);

}

mv::Tensor mv::math::divide(const Tensor& t, float_type x)
{
    dynamic_vector<float_type> data(t.getShape().totalSize());
    for (unsigned i = 0; i < t.getShape().totalSize(); ++i)
        data[i] = t(i) / x;
    return Tensor("div_" + t.getName() + "_scalar", t.getShape(), t.getDType(), t.getOrder(), data);
}

mv::Tensor mv::math::sqrt(const Tensor& t)
{

    dynamic_vector<float_type> data(t.getShape().totalSize());
    for (unsigned i = 0; i < data.size(); ++i)
        data[i] = std::sqrt(t(i));
    return Tensor("sqrt_" + t.getName(), t.getShape(), t.getDType(), t.getOrder(), data);

}