///
/// @file      sipp_api.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for Host side SIPP usage over VPUAL.
///

#ifndef _SIPP_MESSAGES_H_
#define _SIPP_MESSAGES_H_

#include <fstream>
#include <vector>
#include <string.h>
#include <stdint.h>
#include <semaphore.h>
#include "sippDefs.h"

#define MAX_IO (4)
#define MAX_XLINK_CONNECTIONS_FOR_PLUGINS (24)
#define MAX_DMA_FILTERS (48)
#define MAX_CONCURRENT_SIPP_PIPELINES (8)

#define SIPP_ASYNC_CALLBACK_MESSAGE (1)
#define SIPP_ASYNC_PIPELINE_TERMINATE_MESSAGE (2)
#define SIPP_ASYNC_TERMINATE_ALL (-1)
extern "C" {
typedef struct SippAsyncElementType{
    uint64_t pipelineID; /*positive value if it refers to a pipeline ID, negative for SIPP termination*/
    uint32_t message; /*1 if async call should be triggered, 2 if termination should be triggered*/
}SippAsyncMessage;

}

//////////////////////////////////////////////////////////////////
/////////////////////// Pipeline Creation ////////////////////////
//////////////////////////////////////////////////////////////////

class SippPipeline;

typedef void ( * sippEventCallback_t )(
    eSIPP_PIPELINE_EVENT       eEvent,
    SIPP_PIPELINE_EVENT_DATA * ptEventData
);

class SippFilter {
    uint32_t filterID;          // ID of this filter.
    uint32_t nameLength;

    uint64_t pl;           // ID of pipeline to which filter is to be added.
    uint32_t flags;        // Pipeline flags, e.g.SIPP_RESIZE.
    uint32_t outputWidth;  // Width of the frame to be output by the filter.
    uint32_t outputHeight; // Height of the frame to be output by the filter.
    uint32_t numPlanes;    // Number of planes of data in the filter's output buffer.
    uint32_t bpp;          // Bits per pixel of the output buffer data.

    const char *name;      // Character string to identify the filter.

    //Pointer to the pipeline in which this filter resides
    void *pipeline;

public:
    int filter;            // Filter to be created.
    SippFilter() : filterID(0), nameLength(0), pl(0), flags(0), outputWidth(0), outputHeight(0), numPlanes(0), bpp(0), name(nullptr), pipeline(nullptr), filter(0) {
        name != nullptr ? nameLength = strlen(name) : nameLength = 0;
    }

    SippFilter(uint64_t pl, uint32_t flags, uint32_t out_W, uint32_t out_H,
               uint32_t num_pl, uint32_t bpp, int filter, const char* name,
               void* pipeline) : filterID(0), nameLength(0), pl(pl), flags(flags), outputWidth(out_W), outputHeight(out_H),
                                 numPlanes(num_pl), bpp(bpp), name(name),
                                 pipeline(pipeline), filter(filter) {
    }

    ~SippFilter() {
    }

    // Delete copy constructor and copy assignment operator overload.
    SippFilter(const SippFilter&) = delete;
    SippFilter& operator=(const SippFilter&) = delete;

    // Move Constructors.
    SippFilter(SippFilter&&) noexcept {}
    SippFilter& operator=(SippFilter&&) noexcept { return *this; }

    uint32_t getFilterId() const {return filterID;}
    void setFilterId(uint32_t id) { filterID = id;}

    int getFilter() const {return filter;}
    uint32_t getSize() const {return outputWidth * outputHeight * numPlanes * bpp;}
    void *getPipeline() const { return pipeline;}
};

#endif /* _SIPP_MESSAGES_H_ */
