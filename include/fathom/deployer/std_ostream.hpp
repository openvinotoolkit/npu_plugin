#ifndef STD_OSTREAM_HPP_
#define STD_OSTREAM_HPP_

#include <iostream>
#include "include/fathom/deployer/ostream.hpp"

namespace mv
{

    class StdOStream : public OStream
    {

    public:

        StdOStream& operator<<(const string &output)
        {
            std::cout << output;
            return *this;
        }

    };

}

#endif // OSTREAM_HPP_