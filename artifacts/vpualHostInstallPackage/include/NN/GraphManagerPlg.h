///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      GraphManagerPlg.h
/// @copyright All code copyright Movidius Ltd 2018, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for GraphManagerPlg Host FLIC plugin stub using VPUAL.
///
#ifndef __GRAPH_MANAGER_PLG_H__
#define __GRAPH_MANAGER_PLG_H__

// TODO[OB] this is technically not a plugin anymore. Names may be a little missleading.
#include "VpualDispatcher.h"

#include "NN_Types.h"

typedef enum {
    Success,
    No_GraphId_Found,
    Failure,
    Invalid_BlobHndl,
} GraphStatus;

class GraphManagerPlg : public VpualStub {
  public:

    /** Constructor. */
    GraphManagerPlg(uint32_t device_id = 0) : VpualStub("GraphMPDecoder", device_id){};
    // ~GraphManagerPlg();

    void Create();

    GraphStatus NNGraphCheckAvailable(int32_t graphId);

    GraphStatus NNGraphAllocate(BlobHandle_t * Blhdl);
    GraphStatus NNGraphAllocateExistingBlob(BlobHandle_t * Blhdl);

    GraphStatus NNDeallocateGraph(int32_t graphId);
};

#endif // __GRAPH_MANAGER_PLG_H__
