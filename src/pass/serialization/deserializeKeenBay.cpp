#include "include/mcm/pass/serializeKeenBay.hpp"
#include "include/mcm/utils/serializer/fakeGuest.hpp"

#include <chrono>

using namespace MVCNN;

fGraphGuest deserialize(const char* s){
    std::cout << "> Retrieve Header " << std::endl;

    std::ifstream myFile;
    // myFile.open(s);
    myFile.open(s, std::ios::binary | std::ios::in);
    myFile.seekg(0,std::ios::end);
    int length = myFile.tellg();
    myFile.seekg(0,std::ios::beg);
    char *data = new char[length];

    myFile.read(data, length );
    myFile.close();

    fGraphGuest f;
    // in Data

    auto fbb = GetGraphFile(data);

    auto start = std::chrono::high_resolution_clock::now();

    f.version_major = fbb->header()->version()->majorV();
    f.version_minor = fbb->header()->version()->minorV();
    f.version_patch = fbb->header()->version()->patchV();

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";

    return f;

}
