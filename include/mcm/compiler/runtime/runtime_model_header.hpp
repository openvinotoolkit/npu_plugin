#ifndef MV_RUNTIME_MODEL_HEADER_
#define MV_RUNTIME_MODEL_HEADER_

#include <cstdint>
#include <string>
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/runtime_model_link.hpp"
#include "KeemBayFBSchema/compiledSchemas/fileHeader_generated.h"

namespace mv
{
    struct RuntimeModelHeader
    {
        //Version information
        unsigned majorV;
        unsigned minorV;
        unsigned patchV;
        std::string hash;

        //Inputs and outputs
        std::vector<RuntimeModelTensorReference*> * netInput;
        std::vector<RuntimeModelTensorReference*> * netOutput;

        unsigned taskCount;
        unsigned layerCount;

        //Resources
        unsigned shaveMask;
        unsigned nce1Mask;
        unsigned dpuMask;
        unsigned leonCmx;
        unsigned nnCmx;
        unsigned ddrScratch;

        //Network structure
        std::vector<RuntimeModelLink*> * links;
        std::vector<unsigned> * firstID;
    };

    flatbuffers::Offset<MVCNN::SummaryHeader> convertToFlatbuffer(RuntimeModelHeader * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        if(ref)
            return MVCNN::CreateSummaryHeaderDirect(fbb,
                                          MVCNN::CreateVersionDirect(fbb, ref->majorV, ref->minorV, ref->patchV, ref->hash.c_str()),
                                          convertToFlatbuffer(ref->netInput, fbb),
                                          convertToFlatbuffer(ref->netOutput, fbb),
                                          ref->taskCount,
                                          ref->layerCount,
                                          MVCNN::CreateResources(fbb, ref->shaveMask, ref->nce1Mask, ref->dpuMask, ref->leonCmx, ref->nnCmx, ref->ddrScratch),
                                          MVCNN::CreateSourceStructureDirect(fbb, convertToFlatbuffer(ref->links, fbb), ref->firstID));
        else
            return MVCNN::CreateSummaryHeaderDirect(fbb);
    }
}

#endif
