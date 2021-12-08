/*
* {% copyright %}
*/
#ifndef NN_THREAD_H_
#define NN_THREAD_H_

#include <memory>
#include <rtems.h>

namespace nn
{
    namespace util
    {
        class Thread
        {
        public:
            Thread();
            Thread(Thread &&rhs);
            Thread &operator =(Thread &&rhs);
            ~Thread();

            void set_priority(rtems_task_priority prio);
            void set_stack_size(unsigned int size);
            rtems_status_code create(rtems_name name);

            template <typename Arg>
            rtems_status_code start(void (*ep)(Arg *), Arg *arg)
            {
                return start_(reinterpret_cast<rtems_task_entry>(ep), reinterpret_cast<rtems_task_argument>(arg));
            }

            bool joinable() const;
            void join();
            void detach();

            static void yield();
            static void sleep_for(rtems_interval ticks);
            static rtems_task_priority priority();

        private:
            struct ThreadControlBlock;
            std::shared_ptr<ThreadControlBlock> tcb_;

            Thread(const Thread &) = delete;
            Thread &operator =(const Thread &) = delete;

            rtems_status_code start_(rtems_task_entry ep, rtems_task_argument arg);
            rtems_status_code destroy();

            static void thread_func_wrapper(rtems_task_argument arg);
        };
    }
}

#endif // NN_THREAD_H_
