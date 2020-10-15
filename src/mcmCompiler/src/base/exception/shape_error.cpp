#include "include/mcm/base/exception/shape_error.hpp"

mv::ShapeError::ShapeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "ShapeError: " + whatArg)
{

}