#include "mcm/base/order/order.hpp"
#include "mcm/computation/model/types.hpp"
#include "mcm/base/order/col_major.hpp"
#include "mcm/base/order/row_major.hpp"
#include "mcm/base/order/planar.hpp"
#include "mcm/base/order/order_factory.hpp"

std::unique_ptr<mv::OrderClass> mv::OrderFactory::createOrder(mv::Order value)
{
    switch(value)
    {
        case(mv::Order::ColumnMajor):
            return std::unique_ptr<mv::ColMajor>(new ColMajor());
        case(mv::Order::RowMajor):
            return std::unique_ptr<mv::RowMajor>(new RowMajor());
        case(mv::Order::Planar):
            return std::unique_ptr<mv::Planar>(new Planar());
        case(mv::Order::Unknown):
            return std::unique_ptr<mv::ColMajor>(new ColMajor());
    }
}
