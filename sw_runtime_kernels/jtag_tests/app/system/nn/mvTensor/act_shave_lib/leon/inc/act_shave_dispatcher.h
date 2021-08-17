/*
* {% copyright %}
*/
#pragma once

#include <sw_nn_runtime_types.h>

#include <nn_inference_runtime_types.h>
#include <nn_memory.h>
#include <nn_message_queue.h>
#include <nn_relocation.h>
#include <descriptor.h>

#include "sw_layer.h"
//#include "upa_layer_runner.h"

#include <nn_ipc.h>
#include <nn_ipc_config.h>
#include <nn_message_queue.h>
#include <nn_semaphore.h>
#include <nn_thread.h>

#include <memory>
#include <mutex>
#include <list>

#define SHAVE_LAYER_LOAD_ERROR 0

namespace nn {
    namespace act_shave_lib {
        class ACTShaveDispatcher {
        public:
            static std::shared_ptr<ACTShaveDispatcher> getInstance();

            ACTShaveDispatcher(ACTShaveDispatcher &&) = delete;
            ACTShaveDispatcher(const ACTShaveDispatcher &) = delete;
            ACTShaveDispatcher &operator=(ACTShaveDispatcher &&) = delete;
            ACTShaveDispatcher &operator=(const ACTShaveDispatcher &) = delete;

            ACTShaveDispatcher(/**/);
            ~ACTShaveDispatcher();

            /**
             * Registers the SoftChannel Handle with the Dispatcher and initializes
             * the shave_lib backend
             */
            void initSWShaveDispatcher();

            /**
             * Terminate the shave_lib backend
             */
            void terminateSWShaveDispatcher();

            /**
             * Resizes SHAVE pool
             * @param[in] - total_shaves - Number of SHAVEs requested by the inference.
             */
            bool resizeShavePool(unsigned int total_shaves);

            /**
             * Returns true if the minimum resources required to execute a SL are available
             */
            bool hasResources() const;

            /**
             * @returns the SVU shaveID that has taken the role of controller
             */
            unsigned char getControllerShaveID() const;

            /**
             * Flush and invalidate the L2 datacache of all the associated shaves
             */
            void flushShaveL2DataCache();

            /**
             * Invalidate the L2 instruction cache of all the associated shaves
             */
            void flushShaveL2InstructionCache();

            /**
             * The IRS should call this method to give the UPA Dispatcher a soft layer task
             */
            bool enqueueLayerExec(nn::shave_lib::SoftLayerExec *slExec);

            /**
             * Blocks until a soft layer task is completed enqueued by enqueueLayerExec(...)
             * @returns returns the first completed SLE or nullptr if the queue is empty
             */
            nn::shave_lib::SoftLayerExec *dequeueCompletedLayerExec();

        private:
            std::mutex runnerMutex;
            struct ShaveJob{
                nn::shave_lib::SoftLayerExec *sleq = nullptr;
                shv_job_header job_header;
            };
            std::list<ShaveJob> jobs;
            bool shavesStarted = false;
        };
    } // namespace act_shave_lib
} // namespace nn
