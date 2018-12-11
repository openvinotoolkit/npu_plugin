#ifndef EXECUTOR_HPP_
#define EXECUTOR_HPP_

#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"
#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "mvnc.h"


namespace mv
{
    namespace exe
    {
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

            void openDevice(Configuration& configuration);
            void loadGraph(Configuration& configuration);
            void allocateFifos();
            void destroyAll();
            bool checkTargetMatches(Target target, ncDeviceHwVersion_t hwVersion);
        public:
            Tensor execute(Configuration& configuration, Tensor& inputTensor);
            std::string getLogID() const override;
        };
    }
}

#endif // EXECUTOR_HPP_
