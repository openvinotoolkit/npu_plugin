#ifndef SINGLETON_HPP_
#define SINGLETON_HPP_

#include <cstddef>
#include <exception>

namespace mv
{

    namespace base
    {

        class Singleton
        {

            static Singleton *instance_;
            Singleton();
        
        public:

            static Singleton *instance();

        };

    }

}

#endif // SINGLETON_HPP_