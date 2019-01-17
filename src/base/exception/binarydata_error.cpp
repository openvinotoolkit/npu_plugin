#include "include/mcm/base/exception/binarydata_error.hpp"

mv::BinaryDataError::BinaryDataError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "BinaryDataError: " + whatArg)
{

}