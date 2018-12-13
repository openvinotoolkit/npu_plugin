

#include <stdio.h>
#include "contrib/flatbuffers/keenBayFBSchema/compiledSchemas/graphfile_generated.h"
#include <iostream>
#include <fstream>
#include "include/mcm/pass/serializeKeenBay.hpp"
#include "include/mcm/utils/serializer/fakeHost.hpp"
#include "include/mcm/utils/serializer/fakeGuest.hpp"
#include "include/mcm/pass/deserializeKeenBay.hpp"
#include <assert.h>
#include "gtest/gtest.h"



template <typename T>
int EXPECT_EQUALS(T a, T b){
    if(a == b){
        // std::cout < "." << std::endl
        return 0;
    }else{
        std::cout << a << " != " << b << std::endl;
        return 1;
    }
}

TEST (generate_keenbay_blob, test1)
{

    const char * file_name = "VPU3.cpp.bin";

    std::cout << "\n#### GraphFile v3 Tests ####" << std::endl;

    fGraphHost h = fGraphHost();
    flatbuffers::FlatBufferBuilder fbb;

    std::cout << "==== Serialize ====" << std::endl;

    serialize(&h, &fbb, file_name);

    std::cout << "==== DeSerialize ====" << std::endl;

    auto g = deserialize(file_name);

    std::cout << "==== Test ====" << std::endl;

    unsigned fail_count = 0;
    fail_count += EXPECT_EQUALS(g.version_major , h.version[0]);
    fail_count += EXPECT_EQUALS(g.version_minor , h.version[1]);
    fail_count += EXPECT_EQUALS(g.version_patch , h.version[2]);

    if( fail_count )
        std::cout << fail_count << " failed tests." << std::endl;
    else
        std::cout << "All tests passed." << std::endl;

    std::cout << "==== Direct Access Test ====" << std::endl;
    // direct_access(&fbb);

    std::cout << "Unimplemented" << std::endl;
}