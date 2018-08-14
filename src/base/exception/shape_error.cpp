#include "include/mcm/base/exception/shape_error.hpp"

mv::ShapeError::ShapeError(const std::string& whatArg) :
std::logic_error(whatArg)
{

}