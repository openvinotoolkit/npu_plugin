#ifndef MV_ATTRIBUTE_REGISTRY_HPP_
#define MV_ATTRIBUTE_REGISTRY_HPP_

#include <typeinfo>
#include <typeindex>
#include <string>
#include "include/mcm/base/registry.hpp"
#include "include/mcm/base/attribute_entry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    namespace attr
    {
        // TODO Checks for setting type traits
        class AttributeRegistry : public Registry<std::type_index, AttributeEntry>
        {
            
            /**
             * @brief Legal attribute types traits
             */
            static const std::set<std::string> typeTraits_;
            
            /**
             * @brief Legal attribute instances traits
             */
            static const std::set<std::string> instanceTraits_;

        public:

            static AttributeRegistry& instance();
            static bool checkType(const std::type_index& typeID);
            static bool checkType(const std::string& typeName);
            static std::string getTypeName(std::type_index typeID);
            static bool checkValue(std::type_index typeID, const Attribute& val, std::string& msg);
            static const std::function<mv::json::Value(const Attribute&)>& getToJSONFunc(std::type_index typeID);
            static const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc(std::type_index typeID);
            static const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc(std::string typeName);
            static const std::function<std::string(const Attribute&)>& getToStringFunc(std::type_index typeID, bool forceLong = false);
            static std::type_index getTypeID(const std::string& typeName);
            static bool checkTypeTrait(const std::string& typeTrait);
            static bool hasTypeTrait(std::type_index typeID, const std::string& trait);
            static bool checkInstanceTrait(const std::string& trait);

            template <class AttrType>
            static bool checkType()
            {
                return checkType(typeid(AttrType));
            }

            template <class AttrType>
            static std::string getTypeName()
            {
                return getTypeName(typeid(AttrType));
            }

            template <class AttrType>
            AttributeEntry& enter()
            {
                return Registry<std::type_index, AttributeEntry>::enter(typeid(AttrType));
            }

            // Enter or replace if the key is already present in the registry
            template <class AttrType>
            AttributeEntry& enterReplace()
            {
                return Registry<std::type_index, AttributeEntry>::enterReplace(typeid(AttrType));
            }

            template <class AttrType>
            static bool checkValue(const Attribute& val, std::string& msg)
            {
                return checkValue(typeid(AttrType), val, msg);
            }

            inline static const std::function<mv::json::Value(const Attribute&)>& getToJSONFunc(std::type_index typeID)
            {

                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of obtaining to-JSON conversion function for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->getToJSONFunc();
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");
            }

            inline static const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc(std::type_index typeID)
            {
                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of obtaining from-JSON conversion function for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->getFromJSONFunc();
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");
            }

            inline static const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc(std::string typeName)
            {
                return getFromJSONFunc(getTypeID(typeName));
            }

            inline static const std::function<std::string(const Attribute&)>& getToStringFunc(std::type_index typeID)
            {

                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of obtaining to-string conversion function for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->getToStringFunc();
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");

            }

            inline static const std::function<std::vector<uint8_t>(const Attribute&)>& getToBinaryFunc(std::type_index typeID)
            {

                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of obtaining to-string conversion function for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->getToBinaryFunc();
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");

            }

            inline static bool hasTrait(std::type_index typeID, const std::string& trait)
            {
                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of checking type trait for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->hasTrait(trait);
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");
            }

            template <class AttrType>
            static bool hasTypeTrait(const std::string& trait)
            {
                return hasTypeTrait(typeid(AttrType), trait);
            }

        };

        #define STRV(...) #__VA_ARGS__
        #define COMMA ,

        #define MV_REGISTER_ATTR(Type)                                                                          \
            static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =     \
                mv::attr::AttributeRegistry::instance().enter<Type>().setName(STRV(Type))

        #define MV_REGISTER_DUPLICATE_ATTR(Type)                                                                          \
            static ATTRIBUTE_UNUSED(AttributeEntry& CONCATENATE(__ ## AttributeEntry ## __, __COUNTER__)) =     \
                mv::attr::AttributeRegistry::instance().enterReplace<Type>().setName(STRV(Type))


    }

}

#endif // MV_ATTRIBUTE_REGISTRY_HPP_
