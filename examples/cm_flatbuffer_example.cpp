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
    Blob blob("/home/john/vpu_3.blob");
    const auto graph = GetGraphFile(blob.get_ptr());
    deserialize(graph, true);
}