#ifndef MV_ORDER_ERROR_HPP_
#define MV_ORDER_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class OrderError : public std::logic_error
    {

    public:
            
        explicit OrderError(const std::string& whatArg);
        
    };

}

#endif // MV_ORDER_ERROR_HPP_