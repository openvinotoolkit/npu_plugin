#ifndef EXECUTOR_HPP_
#define EXECUTOR_HPP_

#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/target/target_descriptor.hpp"

#include "mvnc.h"


namespace mv
{
    namespace exe
    {
        enum class Protocol
        {
            USB_VSC,
            Unknown
        };

        class Executor : public LogSender
        {
            ncDeviceHandle_t *deviceHandle_;
            ncGraphHandle_t *graphHandle_;
            //Array of buffers to support multi input/output
            struct ncFifoHandle_t ** buffersIn_;
            struct ncFifoHandle_t ** buffersOut_;
            struct ncTensorDescriptor_t* inputTensorDesc_;
            struct ncTensorDescriptor_t* outputTensorDesc_;
            int numInputs_;
            int numOutputs_;
            Target target_;
            Protocol protocol_;

            void openDevice();
            void loadGraph(void* graphFileBuf, int graphLen);
            void allocateFifos();
            void destroyAll();
            Tensor execute_(void* graphFileBuf, int graphLen, Tensor& inputTensor);
            bool checkTargetMatches(ncDeviceHwVersion_t hwVersion);

        public:
            Executor(Target target = Target::Unknown, Protocol protocol = Protocol::USB_VSC);

            Tensor execute(std::shared_ptr<mv::RuntimeBinary> binaryPointer, Tensor& inputTensor);
            Tensor execute(const std::string& graphFilePath, Tensor& inputTensor);

            std::string getLogID() const override;

            void setTarget(Target target);
            void setProtocol(Protocol protocol);

            Target getTarget() const;
            Protocol getProtocol() const;

        };
    }
}

#endif // EXECUTOR_HPP_
