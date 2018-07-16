#ifndef MV_BASE_REGISTRY_HPP_
#define MV_BASE_REGISTRY_HPP_

#include <unordered_map>
#include <memory>
#include <string>

namespace mv
{

    namespace base
    {

        template <class EntryType>
        class Registry
        {

            static std::unordered_map<std::string, EntryType*> reg_;

            Registry()
            {

            }

            Registry(const Registry& other) = delete; 
            Registry& operator=(const Registry& other) = delete;
            ~Registry()
            {
                for (auto it = reg_.begin(); it != reg_.end(); ++it)
                    delete it->second;
            }

        public:

            static Registry& instance();
            inline EntryType& enter(const std::string name)
            {
                EntryType *e = new EntryType(name);
                reg_[name] = e;
                return *e;
            }

        };

        #define ATTRIBUTE_UNUSED __attribute__((unused))

        #define MV_DEFINE_REGISTRY(EntryType)                                           \
            template <class EntryType>                                                  \
            mv::base::Registry<EntryType >& mv::base::Registry<EntryType >::instance()  \
            {                                                                           \
                static Registry instance_;                                              \
                return instance_;                                                       \
            }                                                                           

        #define MV_REGISTER_ENTRY(EntryType, Name)                                      \
            static ATTRIBUTE_UNUSED EntryType& Name =                                   \
                mv::base::Registry<EntryType>::instance().enter(#Name)                  

    }

}

template <class EntryType>
std::unordered_map<std::string, EntryType*> mv::base::Registry<EntryType>::reg_;


#endif // MV_BASE_REGISTRY_HPP_