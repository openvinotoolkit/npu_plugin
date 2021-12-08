/*
* {% copyright %}
*/
#ifndef NN_IPC_H_
#define NN_IPC_H_

#include <Ipc.h>

namespace nn
{
    namespace ipc
    {
        bool init();

        class Channel
        {
        public:
            typedef IpcCallback Callback;

            inline Channel() :
                channel_(nullptr)
            {
            }

            explicit inline Channel(unsigned int id, bool synchronous = true) :
                channel_(nullptr)
            {
                open(id, synchronous);
            }

            inline Channel(unsigned int id, Callback callback, void *context) :
                channel_(nullptr)
            {
                open(id, callback, context);
            }

            inline ~Channel()
            {
                if (is_open())
                    close();
            }

            inline bool is_open() const
            {
                return channel_ != nullptr;
            }

            bool open(unsigned int id, bool synchronous = true);
            bool open(unsigned int id, Callback callback, void *context);

            template <typename T>
            inline bool send(const T& message)
            {
                return send(&message, sizeof(T));
            }

            template <typename T>
            inline bool receive(T*& message)
            {
                return receive(reinterpret_cast<void *&>(message), sizeof(T));
            }

            bool close();

        private:
            IpcChannel *channel_;

            bool send(const void *message, unsigned int size);
            bool receive(void *&message, unsigned int size);

            Channel(const Channel &) = delete;
            Channel &operator =(const Channel &) = delete;
        };
    }
}

#endif /* NN_IPC_H_ */
