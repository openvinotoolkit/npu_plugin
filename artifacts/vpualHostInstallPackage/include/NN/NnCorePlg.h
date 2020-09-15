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
/// @file      NnCorePlg.h
/// @copyright All code copyright Movidius Ltd 2020, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     Header for NnCorePlg Host FLIC plugin using VPUAL.
///
#ifndef __NN_CORE_PLG_H__
#define __NN_CORE_PLG_H__

#include "Flic.h"
#include "Message.h"
#include "NN_Types.h"
#include <vector>

class NnCorePlg : public PluginStub{
  public:
    SReceiver<NnExecMsg>        requestInput;
    MSender<NnExecResponseMsg>  resultOut;

    /** Constructor. */
    NnCorePlg() : PluginStub("NnCorePlg"){};

    MvNCIErrorCode Create(const BlobHandle_t * Blhdl, unsigned int numExecutors);
    MvNCIErrorCode PrepareNetwork();

    unsigned int GetNumberOfInputs() const;
    unsigned int GetNumberOfOutputs() const;
    unsigned int GetScratchBufferSize() const;
    unsigned int GetPrefetchBufferSize() const;
    void SetScratchBuffers(const std::vector<uint32_t> &physAddrs) const;
    void SetPrefetchBuffer(uint32_t physAddr) const;

    MvNCIErrorCode GetBlobVersion(MvNCIVersion *version)  const;
    flicTensorDescriptor_t GetInputTensorDescriptor(unsigned int index) const;
    flicTensorDescriptor_t GetOutputTensorDescriptor(unsigned int index) const;
};
#endif // __NN_CORE_PLG_H__
