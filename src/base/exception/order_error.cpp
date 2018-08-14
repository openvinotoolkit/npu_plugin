#include "include/mcm/base/exception/order_error.hpp"

mv::OrderError::OrderError(const std::string& whatArg) :
std::logic_error(whatArg)
{

}