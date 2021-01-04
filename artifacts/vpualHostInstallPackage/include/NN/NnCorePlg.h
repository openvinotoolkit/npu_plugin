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

/*******************************************************************************
 * @brief FLIC Plugin to perform NN inferences using MvNCI.
 *
 ******************************************************************************/
class NnCorePlg : public PluginStub {
public:
    typedef struct
    {
        uint64_t address;
        std::size_t size;
    } Buffer;
    using BufferVector = std::vector<Buffer>;

    // FLIC messages.
    SReceiver<NnExecMsg>        requestInput; //!< Input inference requests.
    MSender<NnExecResponseMsg>  resultOut;    //!< Output inference results.

    /***************************************************************************
     * @brief Construct the plugin on a specified vpu device.
     *
     * @param device_id    Targeted vpu device-id.
     *
     **************************************************************************/
    NnCorePlg(uint32_t device_id);

    /***************************************************************************
     * @brief Load a network.
     *
     * @param blob_handle   Blob (network) handle.
     * @param num_executors Number of executors to run network on.
     *
     * @return Status of operation.
     *
     **************************************************************************/
    MvNCIErrorCode Create(const BlobHandle_t &blob_handle, unsigned int num_executors);

    /***************************************************************************
     * @brief Prepare a network.
     *
     * Create the executors for the given network. If buffers have been
     * provided to the plugin, the executors will be created with these
     * buffers.
     * Note. This function must be called prior to starting the plugin
     * via the inherited Start() method (typically via Pipeline::Start()).
     *
     * @return Status of operation.
     *
     **************************************************************************/
    MvNCIErrorCode PrepareNetwork(void);

    /***************************************************************************
     * @brief Get the required scratch buffer size for the loaded network.
     *
     * @return Required size of scratch buffer.
     *
     **************************************************************************/
    std::size_t GetScratchBufferSize() const;

    /***************************************************************************
     * @brief Get the required metadata buffer size for the loaded network.
     *
     * @return Required size of metadata buffer.
     *
     **************************************************************************/
    std::size_t GetMetadataBufferSize() const;

    /***************************************************************************
     * @brief Get the required prefetch buffer size for the loaded network.
     *
     * @return Required size of prefetch buffer.
     *
     **************************************************************************/
    std::size_t GetPrefetchBufferSize() const;

    /***************************************************************************
     * @brief Get number of inputs for the loaded network.
     *
     * @return Number of inputs.
     *
     **************************************************************************/
    unsigned int GetNumberOfInputs() const;

    /***************************************************************************
     * @brief Get number of outputs for the loaded network.
     *
     * @return Number of outputs.
     *
     **************************************************************************/
    unsigned int GetNumberOfOutputs() const;

    /***************************************************************************
     * @brief Get the blob version of the loaded network.
     *
     * @param version   Output version info.
     *
     * @return Error if the network isn't loaded.
     *
     **************************************************************************/
    MvNCIErrorCode GetBlobVersion(MvNCIVersion &version) const;

    /***************************************************************************
     * @brief Get the input tensor descriptor at a given index.
     *
     * @param index Index of input tensor to return.
     *
     * @return Pointer to the tensor descriptor.
     *
     **************************************************************************/
    flicTensorDescriptor_t GetInputTensorDescriptor(unsigned int index) const;

    /***************************************************************************
     * @brief Get the output tensor descriptor at a given index.
     *
     * @param index Index of output tensor to return.
     *
     * @return Pointer to the tensor descriptor.
     *
     **************************************************************************/
    flicTensorDescriptor_t GetOutputTensorDescriptor(unsigned int index) const;

    /***************************************************************************
     * @brief Provide scratch buffers for the network to use.
     *
     * Each executor will be prepared with a buffer provided. If the
     * number of buffers provided is less than the number of executors,
     * the MvNCI framework will allocate internal buffers if possible.
     *
     * @param buffers   Vector of scratch buffers to use.
     *
     **************************************************************************/
    void SetScratchBuffers (const BufferVector &buffers);

    /***************************************************************************
     * @brief Provide metadata buffers for the network to use.
     *
     * Each executor will be prepared with a buffer provided. If the
     * number of buffers provided is less than the number of executors,
     * the MvNCI framework will allocate internal buffers if possible.
     *
     * @param buffers   Vector of metadata buffers to use.
     *
     **************************************************************************/
    void SetMetadataBuffers(const BufferVector &buffers);

    /***************************************************************************
     * @brief Provide a prefetch buffer for the network to use.
     *
     * @param buffer    Prefetch buffer.
     *
     **************************************************************************/
    void SetPrefetchBuffer (const Buffer &buffer);

    /***************************************************************************
     * @brief Set the number of UPA shaves for the network to use.
     *
     * @param num_upa_shaves    Number of upa shaves to use.
     *
     **************************************************************************/
    void SetNumUpaShaves(unsigned int num_upa_shaves);
};
#endif // __NN_CORE_PLG_H__
