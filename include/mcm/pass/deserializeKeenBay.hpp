#include <stdio.h>
#include "contrib/flatbuffers/keenBayFBSchema/compiledSchemas/graphfile_generated.h"
#include <iostream>
#include <fstream>
#include "include/mcm/utils/serializer/fakeGuest.hpp"

#ifndef __DESERIALIZE__

using namespace MVCNN;

fGraphGuest deserialize(const char* s);

#define __DESERIALIZE__
#endif