#include <stdio.h>
#include "contrib/flatbuffers/keenBayFBSchema/compiledSchemas/graphfile_generated.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include "include/mcm/pass/serializeKeenBay.hpp"
#include "include/mcm/pass/deserializeKeenBay.hpp"
#include "include/mcm/utils/serializer/fakeGuest.hpp"
#include "include/mcm/utils/serializer/fakeHost.hpp"

int main( int , char**  )
{
    const char * file_name = "VPU3.cpp.bin";

    std::cout << "\n#### GraphFile v3 Tests ####" << std::endl;

    fGraphHost h = fGraphHost();
    flatbuffers::FlatBufferBuilder fbb;

    std::cout << "==== Serialize ====" << std::endl;

    serialize(&h, &fbb, file_name);

    std::cout << "==== DeSerialize ====" << std::endl;

    //auto g = deserialize(file_name);

    return 0;

   
}