#include <stdio.h>
#include "graphfile_generated.h"
#include <iostream>
#include <fstream>
#include "fakeGuest.hpp"

#ifndef __DESERIALIZE__

using namespace MVCNN;

fGraphGuest deserialize(const char* s);

#define __DESERIALIZE__
#endif