#include "include/mcm/tensor/math.hpp"

mv::Tensor mv::math::add(const Tensor& t1, const Tensor& t2)
{

    Tensor output(t1);
    output.add(t2);
    output.setName("add_" + t1.getName() + "_" + t2.getName());
    return output;

}

mv::Tensor mv::math::add(const Tensor& t, double x)
{
    Tensor output(t);
    output.add(x);
    output.setName("add_" + t.getName() + "_scalar");
    return output;
}

mv::Tensor mv::math::subtract(const Tensor& t1, const Tensor& t2)
{

    Tensor output(t1);
    output.subtract(t2);
    output.setName("sub_" + t1.getName() + "_" + t2.getName());
    return output;

}

mv::Tensor mv::math::subtract(const Tensor& t, double x)
{
    Tensor output(t);
    output.subtract(x);
    output.setName("sub_" + t.getName() + "_scalar");
    return output;
}

mv::Tensor mv::math::multiply(const Tensor& t1, const Tensor& t2)
{

    Tensor output(t1);
    output.multiply(t2);
    output.setName("mul_" + t1.getName() + "_" + t2.getName());
    return output;

}

mv::Tensor mv::math::multiply(const Tensor& t, double x)
{
    Tensor output(t);
    output.multiply(x);
    output.setName("mul_" + t.getName() + "_scalar");
    return output;
}

mv::Tensor mv::math::divide(const Tensor& t1, const Tensor& t2)
{

    Tensor output(t1);
    output.divide(t2);
    output.setName("div_" + t1.getName() + "_" + t2.getName());
    return output;

}

mv::Tensor mv::math::divide(const Tensor& t, double x)
{
    Tensor output(t);
    output.divide(x);
    output.setName("div_" + t.getName() + "_scalar");
    return output;
}

mv::Tensor mv::math::sqrt(const Tensor& t)
{
    Tensor output(t);
    output.sqrt();
    output.setName("sqrt_" + t.getName() + "_scalar");
    return output;
}