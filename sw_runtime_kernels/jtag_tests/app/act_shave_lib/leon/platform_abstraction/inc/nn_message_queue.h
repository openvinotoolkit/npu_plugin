/*
* {% copyright %}
*/
#ifndef NN_MESSAGE_QUEUE_H_
#define NN_MESSAGE_QUEUE_H_

#include <rtems.h>
#include <assert.h>

namespace nn
{
    namespace util
    {
        template <typename T>
        class MessageQueue
        {
        public:
            explicit MessageQueue(unsigned int count) :
                id_(0)
            {
                rtems_status_code status = rtems_message_queue_create(
                    reinterpret_cast<rtems_name>(this),
                    count,
                    sizeof(T),
                    RTEMS_FIFO | RTEMS_LOCAL,
                    &id_);
                assert(status == RTEMS_SUCCESSFUL);
                (void)status;
            }

            ~MessageQueue()
            {
                rtems_status_code status = rtems_message_queue_delete(id_);
                assert(status == RTEMS_SUCCESSFUL);
                (void)status;
            }

            bool broadcast(const T &t)
            {
                uint32_t count = 0;
                rtems_status_code status = rtems_message_queue_broadcast(id_, &t, sizeof(T), &count);
                assert(status == RTEMS_SUCCESSFUL);
                return status == RTEMS_SUCCESSFUL;
            }

            bool urgent(const T &t)
            {
                rtems_status_code status = rtems_message_queue_urgent(id_, &t, sizeof(T));
                return status == RTEMS_SUCCESSFUL;
            }

            bool push(const T &t)
            {
                rtems_status_code status = rtems_message_queue_send(id_, &t, sizeof(T));
                return status == RTEMS_SUCCESSFUL;
            }

            bool pop(T &t, unsigned int timeout = RTEMS_NO_TIMEOUT, bool wait = true)
            {
                rtems_option wait_option = wait ? RTEMS_WAIT : RTEMS_NO_WAIT;
                size_t size = 0;
                rtems_status_code status = rtems_message_queue_receive(id_, &t, &size, wait_option, timeout);
                assert(size == sizeof(T) || size == 0);
                assert(status == RTEMS_SUCCESSFUL || status == RTEMS_TIMEOUT || status == RTEMS_UNSATISFIED);
                return status == RTEMS_SUCCESSFUL;
            }

            bool popWithTimeout(T &t, unsigned int timeout)
            {
                size_t size = 0;
                rtems_status_code status = rtems_message_queue_receive(id_, &t, &size, RTEMS_WAIT, timeout);
                assert(size == sizeof(T) || size == 0);
                assert(status == RTEMS_SUCCESSFUL || status == RTEMS_TIMEOUT || status == RTEMS_UNSATISFIED);
                return (status == RTEMS_TIMEOUT);
            }

            unsigned int size() const
            {
                unsigned int count = 0;
                rtems_status_code status = rtems_message_queue_get_number_pending(id_, &count);
                assert(status == RTEMS_SUCCESSFUL);
                return count;
            }

        private:
            rtems_id id_;

            MessageQueue(const MessageQueue &) = delete;
            MessageQueue &operator =(const MessageQueue &) = delete;
        };
    }
}

#endif // NN_MESSAGE_QUEUE_H_
