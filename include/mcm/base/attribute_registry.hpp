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

        class AttributeRegistry : public Registry<std::type_index, AttributeEntry>
        {

            class AttributeRegLogSender : public LogSender
            {

            public:

                std::string getLogID() const override
                {
                    return "AttributeRegistry";
                }

            };

        public:

            static AttributeRegistry& instance();

            inline static bool checkType(const std::type_index& typeID)
            {
                return instance().find(typeID) != nullptr;
            }

            inline static bool checkType(const std::string& typeName)
            {
                return checkType(getTypeID(typeName));
            }

            template <class AttrType>
            inline static bool checkType()
            {
                return checkType(typeid(AttrType));
            }

            inline static std::string getTypeName(std::type_index typeID)
            {

                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of obtaining the name of an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    return attrPtr->getTypeName();
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");

            }

            template <class AttrType>
            inline static std::string getTypeName()
            {
                return getTypeName(typeid(AttrType));
            }

            inline static std::type_index getTypeID(const std::string& typeName)
            {
                for (auto it = instance().reg_.begin(); it != instance().reg_.end(); ++it)
                    if (it->second->getTypeName() == typeName)
                        return it->first;
                throw AttributeError(AttributeRegLogSender(), "Type ID undefined for an unregistered attribute type "
                        + typeName);
            }

            template <class AttrType>
            inline AttributeEntry& enter()
            {
                return Registry<std::type_index, AttributeEntry>::enter(typeid(AttrType));
            }

            // Enter or replace if the key is already present in the registry
            template <class AttrType>
            inline AttributeEntry& enterReplace()
            {
                return Registry<std::type_index, AttributeEntry>::enterReplace(typeid(AttrType));
            }

            inline static bool checkValue(std::type_index typeID, const Attribute& val, std::string& msg)
            {

                if (!checkType(typeID))
                {
                    throw AttributeError(AttributeRegLogSender(), "Attempt of checking the value for an unregistered attribute type "
                        + std::string(typeID.name()));
                }

                AttributeEntry* const attrPtr = instance().find(typeID);

                if (attrPtr)
                {
                    if (!attrPtr->hasCheckFunc())
                        return true;
                    auto fcnPtr = attrPtr->getCheckFunc();
                    return fcnPtr(val, msg);
                }

                throw MasterError(AttributeRegLogSender(), "Registered attribute type " + std::string(typeID.name()) +
                    " not found in the attribute registry");

            }

            template <class AttrType>
            inline static bool checkValue(const Attribute& val, std::string& msg)
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
            inline static bool hasTrait(const std::string& trait)
            {
                return hasTrait(typeid(AttrType), trait);
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
