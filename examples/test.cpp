#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/order.hpp"

int main()
{

    double start = -100.0f;
    double diff = 0.5f;

    mv::Shape tShape({64, 64, 1024});
    std::vector<double> data1 = mv::utils::generateSequence<double>(tShape.totalSize(), start, diff);
    //std::vector<double> data2 = mv::utils::generateSequence<double>(tShape.totalSize(), -start, -diff);

    mv::Tensor t1("t1", tShape, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, data1);
    //mv::Tensor t2("t2", tShape, mv::DTypeType::Float16, mv::OrderType::RowMajor, data2);

    //auto t3 = mv::math::add(t1, t2);

    std::cout << t1.getShape().totalSize() << std::endl;
}