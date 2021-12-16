/*
* {% copyright %}
*/
#include "nn_ipc.h"
#include <nn_log.h>
#include "nn_ipc_config.h"

namespace nn
{
    namespace ipc
    {
        const IpcProcessorId dstProcId = (ipcWhoAmI() == IPC_PROC_LEON_MSS) ? IPC_PROC_LEON_NCE : IPC_PROC_LEON_MSS;

        bool init()
        {
            IpcInitArray myIpcConfig =
            {
                { IPC_PROC_LEON_MSS, IPC_PROC_LEON_NCE, ChannelId::Count },
                IPC_ARRAY_TERMINATOR
            };

            rtems_status_code status = IpcInitialize(myIpcConfig);
            return RTEMS_SUCCESSFUL == status;
        }

        bool Channel::open(unsigned int id, bool synchronous)
        {
            if (is_open())
            {
                nnLog(MVLOG_ERROR, "IPC channel already open");
                return false;
            }

            IpcRxMode ipcRxMode = (synchronous) ? IPC_MODE_BLOCKING : IPC_MODE_ASYNC;
            IpcChannelConfig channel_config = {(uint16_t)id, dstProcId, ipcRxMode, Config::RxBufferLength, nullptr, nullptr};

            rtems_status_code status = IpcCreateChannel(&channel_, &channel_config);
            return RTEMS_SUCCESSFUL == status;
        }

        bool Channel::open(unsigned int id, Callback callback, void *context)
        {
            if (is_open())
            {
                nnLog(MVLOG_ERROR, "IPC channel already open");
                return false;
            }

            IpcChannelConfig channel_config = {(uint16_t)id, dstProcId, IPC_MODE_CALLBACK, Config::RxBufferLength, callback, context};

            rtems_status_code status = IpcCreateChannel(&channel_, &channel_config);
            return RTEMS_SUCCESSFUL == status;
        }

        bool Channel::close()
        {
            if (!is_open())
            {
                nnLog(MVLOG_ERROR, "IPC channel is not open");
                return false;
            }

            rtems_status_code status = IpcCloseChannel(channel_);
            channel_ = nullptr;
            return RTEMS_SUCCESSFUL == status;
        }

        bool Channel::send(const void *message, unsigned int size)
        {
            rtems_status_code status = IpcSendMessage(channel_, const_cast<void *>(message), size);
            return RTEMS_SUCCESSFUL == status;
        }

        bool Channel::receive(void*& message, unsigned int size)
        {
            unsigned int bytes = 0;
            rtems_status_code status = IpcRecvMessage(channel_, &message, reinterpret_cast<uint32_t *>(&bytes));
            return (RTEMS_SUCCESSFUL == status) && (bytes == size);
        }
    }
}
