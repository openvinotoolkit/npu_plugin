#ifndef CSTD_OSTREAM_HPP_
#define CSTD_OSTREAM_HPP_

#include <iostream>
#include "include/fathom/deployer/ostream.hpp"

namespace mv
{

    class CStdOStream : public OStream
    {

    public:

        CStdOStream& operator<<(const string &output)
        {
            std::cout << output;
            return *this;
        }

        bool open()
        {
            return true;
        }

        void close()
        {

        }

    };

}

#endif // COSTREAM_HPP_