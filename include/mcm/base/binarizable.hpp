#ifndef BINARIZABLE_HPP
#define BINARIZABLE_HPP

#include "include/mcm/base/json/json.hpp"

namespace mv
{

    class Binarizable
    {

    public:

        virtual ~Binarizable() = 0;
        virtual std::vector<uint8_t> toBinary() const = 0;

    };

}

#endif
