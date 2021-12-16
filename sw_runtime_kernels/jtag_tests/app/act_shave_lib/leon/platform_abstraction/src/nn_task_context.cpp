/*
* {% copyright %}
*/
#include "nn_task_context.h"
#include <nn_log.h>

namespace nn
{
    namespace util
    {
        TaskContext::TaskContext(rtems_id event) :
            task_(rtems_task_self()),
            event_(event)
        {
        }

        void TaskContext::wait()
        {
            nnLog(MVLOG_DEBUG, "CC task %lu waiting for event %lu...", task_, event_);

            for (rtems_event_set event;
                rtems_event_receive(event_, RTEMS_WAIT | RTEMS_EVENT_ALL, RTEMS_NO_TIMEOUT, &event) != RTEMS_SUCCESSFUL ||
                event != event_;);

            nnLog(MVLOG_DEBUG, "CC task %lu received event %lu...", task_, event_);
        }
    }
}
