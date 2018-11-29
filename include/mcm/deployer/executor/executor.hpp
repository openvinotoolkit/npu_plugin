#ifndef EXECUTOR_HPP_
#define EXECUTOR_HPP_

#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/computation/model/runtime_binary.hpp"
#include "include/mcm/deployer/executor/configuration.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "mvnc.h"


namespace mv
{
    class Executor : public LogSender
    {
        mv::Configuration configuration_;
        ncDeviceHandle_t *deviceHandle_;
        ncGraphHandle_t *graphHandle_;
        //Array of buffers to support multi input/output
        struct ncFifoHandle_t ** buffersIn_;
		struct ncFifoHandle_t ** buffersOut_;
        int numInputs_;
        int numOutputs_;

        void openDevice();
        void loadGraph();
        void allocateFifos();
        void destroyAll();
        bool checkTargetMatches(mv::Target target, ncDeviceHwVersion_t hwVersion);

    public:
        Executor(mv::Configuration& configuration);
        //mv::Tensor execute();
        char* execute();
        std::string getLogID() const override;
        ~Executor();
    };

}

#endif // EXECUTOR_HPP_
