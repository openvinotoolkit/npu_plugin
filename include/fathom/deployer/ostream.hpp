#ifndef OSTREAM_HPP_
#define OSTREAM_HPP_

#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class OStream
    {

    public:

        virtual OStream& operator<<(const string &output) = 0;

    };

}

#endif // OSTREAM_HPP_