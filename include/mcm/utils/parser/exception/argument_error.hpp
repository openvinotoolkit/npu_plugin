#ifndef MV_ARGUMENT_ERROR_HPP_
#define MV_ARGUMENT_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class ArgumentError : public std::runtime_error
    {

        std::string argName_;
        std::string argVal_;

    public:

        explicit ArgumentError(const std::string& argName, const std::string& argVal,
             const std::string& whatArg);
        
        const std::string& getArgName() const;
        const std::string& getArgVal() const;

    };

}

#endif // MV_ARGUMENT_ERROR_HPP_