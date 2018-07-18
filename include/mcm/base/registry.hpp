#ifndef MV_BASE_REGISTRY_HPP_
#define MV_BASE_REGISTRY_HPP_

#include <unordered_map>
#include <string>
#include <vector>

namespace mv
{

    //namespace base
    //{

        template <class EntryType>
        class Registry
        {

            std::unordered_map<std::string, EntryType*> reg_;

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
            inline EntryType& enter(const std::string& name)
            {
                assert(find(name) == nullptr && ("Duplicated registry entry " + name).c_str());
                EntryType *e = new EntryType(name);
                reg_.emplace(name, e);
                return *e;
            }

            inline void remove(const std::string& name)
            {
                assert(find(name) != nullptr && ("Attempt of removal of non-existing entry  " + name).c_str());
                delete reg_[name];
                reg_.erase(name);
            }

            inline bool hasEntry(const std::string& name)
            {
                return find(name) != nullptr;
            }
            
            inline EntryType* find(const std::string& name)
            {
                auto it = reg_.find(name);
                if (it != reg_.end())
                    return &(*it->second);
                return nullptr; 
            }

            inline std::vector<std::string> list()
            {
                std::vector<std::string> result;
                for (auto entry : reg_)
                    result.push_back(entry.first);
                return result;
            }

            inline std::size_t size()
            {
                return reg_.size();
            }

            inline void clear()
            {
                reg_.clear();
            }

        };

        #define ATTRIBUTE_UNUSED __attribute__((unused))

        #define MV_DEFINE_REGISTRY(EntryType)                                           \
            template <class EntryType>                                                  \
            mv::Registry<EntryType >& mv::Registry<EntryType >::instance()  \
            {                                                                           \
                static Registry instance_;                                              \
                return instance_;                                                       \
            }                                                                           

        #define MV_REGISTER_ENTRY(EntryType, Name)                                      \
            static ATTRIBUTE_UNUSED EntryType& __ ## EntryType ## Name ## __ =          \
                mv::Registry<EntryType>::instance().enter(#Name)                 

    //}

}

#endif // MV_BASE_REGISTRY_HPP_