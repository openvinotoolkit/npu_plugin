/*
* {% copyright %}
*/
#ifndef NN_IPC_CONFIG_H_
#define NN_IPC_CONFIG_H_

namespace nn
{
    namespace ipc
    {
        struct Config
        {
            enum
            {
                RxBufferLength = 128,
            };
        };

        struct ChannelId
        {
            enum
            {
                Control,
                Workload,
                Count,
            };
        };
    }
}

#endif // NN_IPC_CONFIG_H_
