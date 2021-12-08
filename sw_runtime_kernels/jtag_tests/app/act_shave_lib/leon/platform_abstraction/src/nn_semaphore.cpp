/*
* {% copyright %}
*/
#include "nn_semaphore.h"
#include <assert.h>

namespace nn
{
    namespace util
    {
        Semaphore::Semaphore(unsigned int count) :
            id_(0)
        {
            rtems_status_code status = rtems_semaphore_create(
                reinterpret_cast<rtems_name>(this),
                count,
                RTEMS_PRIORITY,
                RTEMS_DEFAULT_ATTRIBUTES,
                &id_);
            assert(status == RTEMS_SUCCESSFUL);
            (void)status;
        }

        Semaphore::~Semaphore()
        {
            rtems_status_code status = rtems_semaphore_delete(id_);
            assert(status == RTEMS_SUCCESSFUL);
            (void)status;
        }

        void Semaphore::lock(unsigned int timeout)
        {
            rtems_status_code status = rtems_semaphore_obtain(id_, RTEMS_WAIT, timeout);
            assert(status == RTEMS_SUCCESSFUL || status == RTEMS_TIMEOUT);
            (void)status;
        }

        bool Semaphore::tryLock()
        {
            rtems_status_code status = rtems_semaphore_obtain(id_, RTEMS_NO_WAIT, RTEMS_NO_TIMEOUT);
            assert(status == RTEMS_SUCCESSFUL || status == RTEMS_UNSATISFIED);
            return status == RTEMS_SUCCESSFUL;
        }

        void Semaphore::unlock()
        {
            rtems_status_code status = rtems_semaphore_release(id_);
            assert(status == RTEMS_SUCCESSFUL);
            (void)status;
        }
    }
}
