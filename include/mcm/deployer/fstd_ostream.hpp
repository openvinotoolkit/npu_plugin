#ifndef FSTD_OSTREAM_HPP_
#define FSTD_OSTREAM_HPP_

#include <fstream>
#include "include/mcm/deployer/ostream.hpp"

namespace mv
{

    class FStdOStream : public OStream
    {

        string fileName_;
        std::fstream outStream_;

    public:

        FStdOStream(const string& fileName = "out.txt") :
        fileName_(fileName)
        {

        }

        void setFileName(const string& fileName)
        {
            fileName_ = fileName;
        }

        FStdOStream& operator<<(const string &output)
        {
            outStream_ << output;
            return *this;
        }

        bool open()
        {
            if (outStream_.is_open())
                return false;

            outStream_.open(fileName_, std::ofstream::out);
            return true;
        }

        void close()
        {
            outStream_.close();
        }

    };

}

#endif // FOSTREAM_HPP_