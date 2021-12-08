/*
* {% copyright %}
*/
#ifndef NN_TASK_CONTEXT_H_
#define NN_TASK_CONTEXT_H_

#include <rtems.h>

namespace nn
{
    namespace util
    {
        struct TaskContext
        {
            TaskContext(rtems_id event);

            inline void remap()
            {
                task_ = rtems_task_self();
            }

            void wait();

            inline void notify()
            {
                rtems_event_send(task_, event_);
            }

        private:
            rtems_id task_;
            rtems_id event_;
        };
    }
}

#endif // NN_TASK_CONTEXT_H_
