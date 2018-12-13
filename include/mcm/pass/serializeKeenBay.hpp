#include <stdio.h>
#include "contrib/flatbuffers/keenBayFBSchema/compiledSchemas/graphfile_generated.h"
#include "contrib/flatbuffers/keenBayFBSchema/compiledSchemas/nnDPU_generated.h"
#include <iostream>
#include <fstream>
#include "include/mcm/utils/serializer/fakeHost.hpp"

#ifndef __SERIALIZE__

using namespace MVCNN;

void serialize(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb, const char* s);
flatbuffers::Offset<SummaryHeader> getHeader(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);
flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MVCNN::TaskList> > > getTaskLists(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);


#define __SERIALIZE__
#endif