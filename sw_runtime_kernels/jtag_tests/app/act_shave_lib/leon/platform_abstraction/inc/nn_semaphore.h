/*
* {% copyright %}
*/
#ifndef NN_SEMAPHORE_H
#define NN_SEMAPHORE_H

#include <rtems.h>

namespace nn
{
    namespace util
    {
        class Semaphore
        {
        public:
            explicit Semaphore(unsigned int count = 1);
            ~Semaphore();

            void lock(unsigned int timeout = 0);
            bool tryLock();
            void unlock();

        private:
            rtems_id id_;

            Semaphore(const Semaphore &) = delete;
            Semaphore &operator =(const Semaphore &) = delete;
        };

        class SemaphoreLocker
        {
        public:
            SemaphoreLocker(Semaphore &semaphore) :
                semaphore_(&semaphore)
            {
                semaphore.lock();
            }

            void unlock()
            {
                if (semaphore_ != nullptr)
                    semaphore_->unlock(), semaphore_ = nullptr;
            }

            ~SemaphoreLocker()
            {
                unlock();
            }

        private:
            Semaphore *semaphore_;

            SemaphoreLocker(const SemaphoreLocker &) = delete;
            SemaphoreLocker &operator =(const SemaphoreLocker &) = delete;
        };
    }
}

#endif //NN_SEMAPHORE_H
