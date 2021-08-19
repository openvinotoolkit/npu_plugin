// {% copyright %}

#include "CustomCpp.h"
#include <mv_types.h>
#include <nn_log.h>

#include "upa_task_runner.hpp"

using namespace mv::tensor;

CustomCpp::~CustomCpp() = default;

void CustomCpp::run(mv::tensor::Processor& ,
                  t_MvTensorMyriadResources& myriadRes,
                  t_MvTensorDebugInfo&) {
    for(const Buffer& input: inputVec) {
        nnLog(MVLOG_DEBUG, "ndims  = %ld\n", input.ndims);
        nnLog(MVLOG_DEBUG, "order  = 0x%x\n", input.order);
        nnLog(MVLOG_DEBUG, "dims.0 = %ld(%ld)\n", input.dims[0], input.strides[0]);
        nnLog(MVLOG_DEBUG, "dims.1 = %ld(%ld)\n", input.dims[1], input.strides[1]);
        nnLog(MVLOG_DEBUG, "dims.2 = %ld(%ld)\n", input.dims[2], input.strides[2]);
        nnLog(MVLOG_DEBUG, "dims.3 = %ld(%ld)\n", input.dims[3], input.strides[3]);
        nnLog(MVLOG_DEBUG, "Input buffer %p.\n", input.addr);
    }
    for(const Buffer& output: outputVec) {
        nnLog(MVLOG_DEBUG, "Output buffer %p.\n", output.addr);
    }
    nnLog(MVLOG_DEBUG, "leonPreambleID %ld\n", ops.leonPreambleID);
    nnLog(MVLOG_DEBUG, "KernelData %p with length %ld.\n", ops.kernelData, ops.kernelDataLen);
    nnLog(MVLOG_DEBUG, "ParamData  %p with length %ld.\n", ops.paramData,  ops.paramDataLen);

    std::vector<uint64_t> kernelData((uint64_t*)ops.kernelData, (uint64_t*)ops.kernelData+(ops.kernelDataLen+7)/8);
    std::unique_ptr<MVCNN::BinaryDataT> kbdt(new MVCNN::BinaryDataT);
    kbdt->data = kernelData;
    kbdt->underlying_type = MVCNN::DType_U8;
    kbdt->length = ops.kernelDataLen;

    int byteDataLenght = sizeof(uint32_t)/*Common arg buffer size*/ +
                         sizeof(uint32_t)/*Op ID*/ +
                         sizeof(uint32_t)/*offset to InOuts*/ +
                         ops.paramDataLen * sizeof(uint32_t) +
                         sizeof(uint32_t)/*Num inputs*/ +
                         sizeof(uint32_t)/*Num outputs*/ +
                         inputVec.size() * sizeof(nn::TensorRefNDData) +
                         outputVec.size() * sizeof(nn::TensorRefNDData);
//    std::vector<uint64_t> paramData((uint64_t*)ops.paramData, (uint64_t*)ops.paramData+(ops.paramDataLen+1)/2);
    std::vector<uint64_t> paramData((byteDataLenght + 7) / 8);
    nnLog(MVLOG_DEBUG, "byteDataLenght  = %d, paramData.size %d\n", byteDataLenght, paramData.size());
    uint32_t* paramDataBuffer = reinterpret_cast<uint32_t*>(paramData.data());
    paramDataBuffer[0] = paramData.size() * 2; // size in uin32_t elements
    nnLog(MVLOG_DEBUG, "paramDataBuffer serialization1 ops.opID %d\n", ops.opID);
    paramDataBuffer[1] = (2 + ops.paramDataLen) * sizeof(uint32_t); // Size of all parameters in bytes
    paramDataBuffer[2] = ops.opID;
    memcpy_s(paramDataBuffer + 3, ops.paramDataLen * sizeof(uint32_t), ops.paramData, ops.paramDataLen * sizeof(uint32_t));
    uint32_t* paramInOutBuffer = paramDataBuffer + 3 + ops.paramDataLen;
    paramInOutBuffer[0] = inputVec.size();
    paramInOutBuffer[1] = outputVec.size();
    uint8_t* inOutBuffer = (uint8_t*)(paramInOutBuffer + 2);
    for (unsigned i = 0; i < inputVec.size(); i++) {
        memcpy_s(inOutBuffer, sizeof(nn::TensorRefNDData), &(inputVec[i]), sizeof(nn::TensorRefNDData));
        inOutBuffer += sizeof(nn::TensorRefNDData);
    }
    for (unsigned i = 0; i < outputVec.size(); i++) {
        memcpy_s(inOutBuffer, sizeof(nn::TensorRefNDData), &(outputVec[i]), sizeof(nn::TensorRefNDData));
        inOutBuffer += sizeof(nn::TensorRefNDData);
    }

    std::unique_ptr<MVCNN::BinaryDataT> pbdt(new MVCNN::BinaryDataT);
    pbdt->data = paramData;
    pbdt->underlying_type = MVCNN::DType_U32;
    pbdt->length = ops.paramDataLen;

    // ---- KMB infra
    MVCNN::CustomLayerCppParamsT* softLayerParamsValue = new MVCNN::CustomLayerCppParamsT;
    softLayerParamsValue->leonPreambleID = ops.leonPreambleID;
    softLayerParamsValue->kernelData = std::move(kbdt);
    softLayerParamsValue->paramData = std::move(pbdt);

    std::unique_ptr<MVCNN::UPALayerTaskT> upaTask (new MVCNN::UPALayerTaskT());
    upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_CustomLayerCppParams;
    upaTask->softLayerParams.value = softLayerParamsValue;

    UPATaskRunner runner;
    mvTensorAssert(runner.enqueTask(std::move(upaTask), std::move(inputVec), std::move(outputVec), myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "custom OpenCL layer run failed");
    mvTensorAssert(runner.dequeResult(), "custom Cpp layer run failed");
}
