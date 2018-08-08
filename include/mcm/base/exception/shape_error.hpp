#ifndef MV_SHAPE_ERROR_HPP_
#define MV_SHAPE_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class ShapeError : public std::logic_error
    {

    public:

        explicit ShapeError(const std::string& whatArg);
        
    };

}

#endif // MV_SHAPE_ERROR_HPP_