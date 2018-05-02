#ifndef ALLOCATOR_HPP_
#define ALLOCATOR_HPP_

#include <cassert>

namespace mv
{

    class base_allocator
    {

    private:

        static void default_alloc_fail(int err, char *msg, unsigned len);

    public:
    
        typedef void (*callback)(int err, char *msg, unsigned len);
        
        virtual ~base_allocator() = 0;

        static callback alloc_fail_callback;

    };

}

#endif // ALLOCATOR_HPP_