#ifndef MV_BASE_REGISTRY_HPP_
#define MV_BASE_REGISTRY_HPP_

#include <map>
#include <string>
#include <vector>
#include <cassert>

namespace mv
{

    //namespace base
    //{

        template <class RegistryType, class KeyType, class EntryType>
        class Registry
        {
            
        protected:

            std::map<KeyType, EntryType*> reg_;

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

            static RegistryType& instance();

            inline EntryType& enter(const KeyType& key)
            {
                assert(find(key) == nullptr && "Duplicated registry entry");
                EntryType *e = new EntryType(key);
                reg_.emplace(key, e);
                return *e;
            }

            inline EntryType& enterReplace(const KeyType& key)
            {
                if (find(key) != nullptr)
                    remove(key);
                EntryType *e = new EntryType(key);
                reg_.emplace(key, e);
                return *e;
            }

            inline void remove(const KeyType& key)
            {
                assert(find(key) != nullptr && "Attempt of removal of non-existing entry");
                delete reg_[key];
                reg_.erase(key);
            }

            inline bool hasEntry(const KeyType& key)
            {
                return find(key) != nullptr;
            }
            
            inline EntryType* find(const KeyType& key)
            {
                auto it = reg_.find(key);
                if (it != reg_.end())
                    return &(*it->second);
                return nullptr; 
            }

            inline std::vector<KeyType> list()
            {
                std::vector<KeyType> result;
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

		#ifdef ATTRIBUTE_UNUSED
		#elif defined(__GNUC__)
			# define ATTRIBUTE_UNUSED(x) x __attribute__((unused))
		#else
			# define ATTRIBUTE_UNUSED(x) x
		#endif

        #define MV_DEFINE_REGISTRY(RegistryType, KeyType, EntryType)                        \
            template <>                                                                     \
            RegistryType& mv::Registry<RegistryType, KeyType, EntryType >::instance()       \
            {                                                                               \
                static RegistryType instance_;                                              \
                return instance_;                                                           \
            }                                                                               
               

        #define CONCATENATE_DETAIL(x, y) x##y
        #define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)                                                      

        #define MV_REGISTER_ENTRY(RegistryType, KeyType, EntryType, key)                            \
            static ATTRIBUTE_UNUSED(EntryType& CONCATENATE(__ ## EntryType ## __, __COUNTER__)) =   \
                mv::Registry<RegistryType, KeyType, EntryType >::instance().enter(key)                 

    //}

}

#endif // MV_BASE_REGISTRY_HPP_
