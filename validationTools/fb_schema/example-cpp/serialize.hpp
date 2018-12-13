#include <stdio.h>
#include "graphfile_generated.h"
#include "nnNCE2_generated.h"
#include <iostream>
#include <fstream>
#include "fakeHost.hpp"

#ifndef __SERIALIZE__

using namespace MVCNN;

void serialize(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb, const char* s);
flatbuffers::Offset<SummaryHeader> getHeader(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);
flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MVCNN::TaskList> > > getTaskLists(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);


flatbuffers::Offset<SummaryHeader> getHeader(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);
flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<MVCNN::TaskList> > >  getTaskLists(fGraphHost * fg, flatbuffers::FlatBufferBuilder * fbb);

#define __SERIALIZE__
#endif