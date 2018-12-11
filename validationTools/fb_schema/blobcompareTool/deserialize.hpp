#include <stdio.h>
#include <iostream>
#include <fstream>

#include "graphfile_generated.h"

#ifndef __DESERIALIZE__

using namespace MVCNN;

class Blob
{
private:
    char* data;

public:
    Blob(const char* s)
    {
        std::ifstream myFile;

        myFile.open(s, std::ios::binary | std::ios::in);
        myFile.seekg(0,std::ios::end);
        int length = myFile.tellg();
        myFile.seekg(0,std::ios::beg);

        std::cout << "Reading " << length << " bytes from "
                  << s << std::endl;

        data = new char[length];
        assert(data != nullptr);

        myFile.read(data, length );
        myFile.close();
    }
    Blob(const Blob&) = delete;

    ~Blob()
    {
        if (data != nullptr)
            delete data;
    }

    char* get_ptr()
    {
        return data;
    }
};

void deserialize(const GraphFile* const graph, bool print);

#define __DESERIALIZE__
#endif
