/*
* {% copyright %}
*/
#include "nn_thread.h"
#include "nn_semaphore.h"
#include <assert.h>
#include <algorithm>

namespace nn
{
    namespace util
    {
        struct Thread::ThreadControlBlock
        {
            rtems_id id_;
            rtems_task_entry user_ep_;
            rtems_task_argument user_arg_;
            unsigned int priority_;
            unsigned int stack_size_;
            Semaphore joined_;

            ThreadControlBlock() :
                id_(RTEMS_INVALID_ID),
                user_ep_(nullptr),
                user_arg_(0),
                priority_(Thread::priority()),
                stack_size_(RTEMS_MINIMUM_STACK_SIZE),
                joined_(0)
            {
            }
        };

        Thread::Thread() :
            tcb_(new ThreadControlBlock)
        {
        }

        Thread::Thread(Thread &&rhs) :
            tcb_()
        {
            std::swap(tcb_, rhs.tcb_);
        }

        Thread &Thread::operator =(Thread &&rhs)
        {
            std::swap(tcb_, rhs.tcb_);
            return *this;
        }

        Thread::~Thread()
        {
            destroy();
        }

        void Thread::set_priority(rtems_task_priority prio)
        {
            tcb_->priority_ = std::max<rtems_task_priority>(1, prio);
        }

        void Thread::set_stack_size(unsigned int size)
        {
            tcb_->stack_size_ = std::max<unsigned int>(RTEMS_MINIMUM_STACK_SIZE, size);
        }

        rtems_status_code Thread::create(rtems_name name)
        {
            rtems_status_code code = rtems_task_create(
                name,
                tcb_->priority_,
                tcb_->stack_size_,
                RTEMS_DEFAULT_MODES,
                RTEMS_DEFAULT_ATTRIBUTES,
                &tcb_->id_
            );

            // RTEMS recommends to explicitly configure the scheduler of the new thread
            if (code == RTEMS_SUCCESSFUL)
            {
                rtems_id scheduler_id = RTEMS_INVALID_ID;

                code = rtems_task_get_scheduler(
                    rtems_task_self(),
                    &scheduler_id
                );

                if (code == RTEMS_SUCCESSFUL)
                {
                    code = rtems_task_set_scheduler(
                        tcb_->id_,
                        scheduler_id,
                        tcb_->priority_
                    );
                }
            }

            assert(code == RTEMS_SUCCESSFUL && "Could not create RTEMS task");
            return code;
        }

        rtems_status_code Thread::start_(rtems_task_entry ep, rtems_task_argument arg)
        {
            assert(tcb_ && "Trying to start an uncreated thread");

            tcb_->user_ep_ = ep;
            tcb_->user_arg_ = arg;

            rtems_status_code code = rtems_task_start(
                tcb_->id_,
                &Thread::thread_func_wrapper,
                reinterpret_cast<rtems_task_argument>(&tcb_)
            );

            assert(code == RTEMS_SUCCESSFUL && "Could not start RTEMS task");
            return code;
        }

        bool Thread::joinable() const
        {
            return tcb_->id_ != RTEMS_INVALID_ID && tcb_->user_ep_ != nullptr;
        }

        void Thread::join()
        {
            tcb_->joined_.lock();

            // Cannot properly join RTEMS threads, like STD ones.
            // Once the thread function returns the scheduler breaks and freezes all other threads.
            // As a workaround, detach the child and let it destroy itself once its user-code is over.
            detach();
            yield();
        }

        void Thread::detach()
        {
            tcb_.reset(new ThreadControlBlock);
        }

        rtems_status_code Thread::destroy()
        {
            rtems_status_code code = rtems_task_delete(tcb_->id_);
            tcb_->id_ = RTEMS_INVALID_ID;
            return code;
        }

        void Thread::yield()
        {
            rtems_task_wake_after(RTEMS_YIELD_PROCESSOR);
        }

        void Thread::sleep_for(rtems_interval ticks)
        {
            rtems_task_wake_after(ticks);
        }

        rtems_task_priority Thread::priority()
        {
            rtems_task_priority prio = 127;
            rtems_id scheduler_id = RTEMS_INVALID_ID;

            rtems_status_code code = rtems_task_get_scheduler(
                rtems_task_self(),
                &scheduler_id
            );

            if (code == RTEMS_SUCCESSFUL)
            {
                code = rtems_task_get_priority(
                    rtems_task_self(),
                    scheduler_id,
                    &prio
                );
            }

            return prio;
        }

        void Thread::thread_func_wrapper(rtems_task_argument arg)
        {
            if (std::shared_ptr<ThreadControlBlock> *tcb_ptr = reinterpret_cast<std::shared_ptr<ThreadControlBlock> *>(arg))
            {
                // Make sure the new thread hold a reference
                std::shared_ptr<ThreadControlBlock> tcb(*tcb_ptr);

                if (tcb->user_ep_)
                    tcb->user_ep_(tcb->user_arg_);

                tcb->joined_.unlock();
            }

            // See comment in Thread::join()
            rtems_task_delete(RTEMS_SELF);
        }
    }
}
