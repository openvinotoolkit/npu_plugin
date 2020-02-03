//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <Inference.h>
#include <hddl_unite/hddl2_infer_data.h>

#include <string>

#include "hddl2_remote_blob.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static RemoteMemoryFD getFDFromRemoteBlob(const IE::Blob::Ptr& blob) {
    RemoteMemoryFD memoryFd = 0;
    try {
        HDDL2RemoteBlob::Ptr remoteBlobPtr = std::dynamic_pointer_cast<HDDL2RemoteBlob>(blob);
        memoryFd = remoteBlobPtr->getRemoteMemoryFD();
    } catch (const std::exception& ex) {
        printf("Failed to get memory fd from remote blob! %s\n", ex.what());
    }
    return memoryFd;
}

HddlUnite::Inference::Precision HddlUniteInferData::convertIEPrecision(const IE::Precision& precision) {
    switch (precision) {
    case IE::Precision::UNSPECIFIED:
        return HddlUnite::Inference::UNSPECIFIED;
    case IE::Precision::MIXED:
        return HddlUnite::Inference::MIXED;
    case IE::Precision::FP32:
        return HddlUnite::Inference::FP32;
    case IE::Precision::FP16:
        return HddlUnite::Inference::FP16;
    case IE::Precision::Q78:
        return HddlUnite::Inference::Q78;
    case IE::Precision::I16:
        return HddlUnite::Inference::I16;
    case IE::Precision::U8:
        return HddlUnite::Inference::U8;
    case IE::Precision::I8:
        return HddlUnite::Inference::I8;
    case IE::Precision::U16:
        return HddlUnite::Inference::U16;
    case IE::Precision::I32:
        return HddlUnite::Inference::I32;
    case IE::Precision::I64:
        return HddlUnite::Inference::I64;
    case IE::Precision::BIN:
        return HddlUnite::Inference::BIN;
    case IE::Precision::CUSTOM:
        return HddlUnite::Inference::CUSTOM;
    default:
        THROW_IE_EXCEPTION << "Incorrect precision";
    }
}

static void checkInputArguments(const std::string& inputName, const IE::Blob::Ptr& blob) {
    if (inputName.empty()) {
        THROW_IE_EXCEPTION << "Name is empty";
    }
    if (blob == nullptr) {
        THROW_IE_EXCEPTION << "Blob is null";
    }
    if (blob->size() == 0) {
        THROW_IE_EXCEPTION << "Blob is empty";
    }
}

//------------------------------------------------------------------------------
//      class HddlUniteInferData Implementation
//------------------------------------------------------------------------------
HddlUniteInferData::HddlUniteInferData(const HDDL2RemoteContext::Ptr& remoteContext) {
    _auxBlob = {HddlUnite::Inference::AuxBlob::Type::TimeTaken};

    HddlUnite::WorkloadContext::Ptr workloadContext = nullptr;
    if (remoteContext != nullptr) {
        workloadContext = remoteContext->getHddlUniteWorkloadContext();
    }

    if (workloadContext != nullptr) {
        isVideoWorkload = true;
        _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob, workloadContext);
    } else {
        _inferDataPtr = HddlUnite::Inference::makeInferData(_auxBlob);
    }

    if (_inferDataPtr.get() == nullptr) {
        THROW_IE_EXCEPTION << "Failed to create inferData";
    }
}

void HddlUniteInferData::createRemoteDesc(const bool isInput, const std::string& name, const IE::Blob::Ptr& blob) {
    const RemoteMemoryFD remoteMemoryFD = getFDFromRemoteBlob(blob);
    const IE::TensorDesc tensorDesc = blob->getTensorDesc();
    const size_t blobSize = blob->byteSize();

    if (remoteMemoryFD == 0) {
        THROW_IE_EXCEPTION << "Incorrect remote memory file descriptor";
    }
    HddlUnite::Inference::Precision precision = convertIEPrecision(tensorDesc.getPrecision());
    const bool isRemoteMem = true;
    const bool needAllocate = false;

    HddlUnite::Inference::BlobDesc blobDesc(precision, isRemoteMem, needAllocate, blobSize);

    /** @note We will not provide pointer to data (src_ptr) as all synchronization logic hidden in
     *  remote allocator. Only wrap memory. */
    blobDesc.m_fd = remoteMemoryFD;

    _inferDataPtr->createBlob(name, blobDesc, isInput);
}

void HddlUniteInferData::createLocalDesc(const bool isInput, const std::string& name, const IE::Blob::Ptr& blob) {
    const IE::TensorDesc tensorDesc = blob->getTensorDesc();
    const size_t blobSize = blob->byteSize();
    auto localBuffer = blob->buffer().as<void*>();

    HddlUnite::Inference::Precision precision = convertIEPrecision(tensorDesc.getPrecision());

    bool isRemoteMem = false;
    bool needAllocate = false;

    if (isVideoWorkload) {
        /** @note In case of video workload, we should always use remote memory.
         *  For input,  srcPtr will be memory, which will be synced with remote
         *  For output, srcPtr will be memory, to which result will be synced after inference. */
        isRemoteMem = true;
        needAllocate = true;
    }

    HddlUnite::Inference::BlobDesc blobDesc(precision, isRemoteMem, needAllocate, blobSize);
    blobDesc.m_srcPtr = localBuffer;

    _inferDataPtr->createBlob(name, blobDesc, isInput);
}

void HddlUniteInferData::prepareInput(const std::string& inputName, const IE::Blob::Ptr& blob) {
    checkInputArguments(inputName, blob);

    const bool isInputBlob = true;

    if (blob->is<HDDL2RemoteBlob>()) {
        createRemoteDesc(isInputBlob, inputName, blob);
    } else {
        createLocalDesc(isInputBlob, inputName, blob);
    }
}

void HddlUniteInferData::prepareOutput(const std::string& outputName, const IE::Blob::Ptr& blob) {
    checkInputArguments(outputName, blob);

    const bool isInputBlob = false;

    if (blob->is<HDDL2RemoteBlob>()) {
        createRemoteDesc(isInputBlob, outputName, blob);
    } else {
        createLocalDesc(isInputBlob, outputName, blob);
    }
}
