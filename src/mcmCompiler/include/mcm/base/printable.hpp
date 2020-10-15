#ifndef PRINTABLE_HPP_
#define PRINTABLE_HPP_

#include <string>

namespace mv
{

    class Printable
    {

    public:

        virtual ~Printable() = 0;
        static void replaceSub(std::string &input, const std::string &oldSub, const std::string &newSub);
        virtual std::string toString() const = 0;

    };

}

#endif
