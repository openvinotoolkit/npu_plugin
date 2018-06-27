#include "include/mcm/base/singleton.hpp"

mv::base::Singleton *mv::base::Singleton::instance_ = nullptr;

mv::base::Singleton::Singleton()
{

}

mv::base::Singleton *mv::base::Singleton::instance()
{
    
    if (instance_ == nullptr)
        instance_ = new Singleton();

    // TODO Throw exception

    return instance_;

}