#ifndef MV_ATTRIBUTE_ENTRY_HPP_
#define MV_ATTRIBUTE_ENTRY_HPP_

#include <string>
#include <functional>
#include <set>
#include <map>
#include <typeinfo>
#include <typeindex>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/exception/master_error.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/printable.hpp"

namespace mv
{

    class Attribute;

    namespace attr
    {

        class AttributeEntry : public LogSender
        {

            struct GenericFunc
            {
                virtual ~GenericFunc()
                {

                }
            };

            template<class R, class... Args>
            struct ConcreteFunc : GenericFunc
            {
                std::function<R(Args...)> f;
            };

            std::type_index typeID_;
            std::string typeName_;
            std::string description_;
            std::shared_ptr<GenericFunc> checkFunc_;
            std::shared_ptr<GenericFunc> toJSONFunc_;
            std::shared_ptr<GenericFunc> fromJSONFunc_;
            std::shared_ptr<GenericFunc> toStringFunc_;
            std::shared_ptr<GenericFunc> toBinaryFunc_;
            std::set<std::string> typeTraits_;

        public:

            AttributeEntry(const std::type_index& typeID) :
            typeID_(typeID),
            typeName_("UNNAMED")
            {

            }

            inline AttributeEntry& setName(const std::string& typeName)
            {
                typeName_ = typeName;
                Printable::replaceSub(typeName_, " , ", ", ");
                return *this;
            }

            inline const std::string& getTypeName() const
            {
                return typeName_;
            }

            inline std::type_index getTypeID() const
            {
                return typeID_;
            }

            inline AttributeEntry& setDescription(const std::string& description)
            {
                description_ = description;
                return *this;
            }

            inline const std::string getDescription() const
            {
                return description_;
            }

            inline AttributeEntry& setCheckFunc(const std::function<bool(const Attribute&, std::string&)>& f)
            {
                auto pFunc = std::make_shared<ConcreteFunc<bool, const Attribute&, std::string&> >();
                pFunc->f = f;
                checkFunc_ = pFunc;
                return *this;
            }

            inline const std::function<bool(const Attribute&, std::string&)>& getCheckFunc()
            {

                if (!checkFunc_)
                    throw MasterError(*this, "Undefined check function for the argument type " + typeName_);

                auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<bool, const Attribute&, std::string&> >(checkFunc_);
                if (pFunc)
                    return pFunc->f;

                throw AttributeError(*this, "Invalid types specified for check function for type " + typeName_);

            }

            inline bool hasCheckFunc() const
            {
                return static_cast<bool>(checkFunc_);
            }

            inline AttributeEntry& setToJSONFunc(const std::function<mv::json::Value(const Attribute&)>& f)
            {
                auto pFunc = std::make_shared<ConcreteFunc<mv::json::Value, const Attribute&> >();
                pFunc->f = f;
                toJSONFunc_ = pFunc;
                return *this;
            }

            inline AttributeEntry& setToBinaryFunc(const std::function<std::vector<uint8_t>(const Attribute&)>& f)
            {
                auto pFunc = std::make_shared<ConcreteFunc<std::vector<uint8_t>, const Attribute&>>();
                pFunc->f = f;
                toBinaryFunc_ = pFunc;
                return *this;
            }

            inline const std::function<mv::json::Value(const Attribute&)>& getToJSONFunc()
            {

                if (!toJSONFunc_)
                    throw MasterError(*this, "Undefined to-JSON conversion function for the argument type " + typeName_);

                auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<mv::json::Value, const Attribute&> >(toJSONFunc_);
                if (pFunc)
                    return pFunc->f;

                throw AttributeError(*this, "Invalid types specified for to-JSON conversion function for type " + typeName_);

            }

            inline AttributeEntry& setFromJSONFunc(const std::function<Attribute(const mv::json::Value&)>& f)
            {

                auto pFunc = std::make_shared<ConcreteFunc<Attribute, const mv::json::Value&> >();
                pFunc->f = f;
                fromJSONFunc_ = pFunc;

                return *this;

            }

            inline const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc()
            {

                if (!fromJSONFunc_)
                    throw MasterError(*this, "Undefined from-JSON conversion function for the argument type " + typeName_);

                auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<Attribute, const mv::json::Value&> >(fromJSONFunc_);

                if (pFunc)
                    return pFunc->f;

                throw AttributeError(*this, "Invalid types specified for from-JSON conversion function for type " + typeName_);

            }

            inline AttributeEntry& setToStringFunc(const std::function<std::string(const Attribute&)>& f)
            {
                auto pFunc = std::make_shared<ConcreteFunc<std::string, const Attribute&> >();
                pFunc->f = f;
                toStringFunc_ = pFunc;
                return *this;
            }

            inline const std::function<std::string(const Attribute&)>& getToStringFunc()
            {

                if (!toStringFunc_)
                    throw MasterError(*this, "Undefined to-string conversion function for the argument type ");

                auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<std::string, const Attribute&> >(toStringFunc_);
                if (pFunc)
                    return pFunc->f;

                throw AttributeError(*this, "Invalid types specified for to-string conversion function for type " + typeName_);

            }

            inline const std::function<std::vector<uint8_t>(const Attribute&)>& getToBinaryFunc()
            {

                if (!toBinaryFunc_)
                    throw MasterError(*this, "Undefined to-binary conversion function for the argument type ");

                auto pFunc = std::dynamic_pointer_cast<ConcreteFunc<std::vector<uint8_t>, const Attribute&> >(toBinaryFunc_);
                if (pFunc)
                    return pFunc->f;

                throw AttributeError(*this, "Invalid types specified for to-binary conversion function for type " + typeName_);

            }

            inline AttributeEntry& setTrait(const std::string& trait)
            {
                if (typeTraits_.find(trait) != typeTraits_.end())
                    typeTraits_.insert(trait);
                return *this;
            }

            inline bool hasTrait(const std::string& trait)
            {
                return typeTraits_.find(trait) != typeTraits_.end();
            }

            std::string getLogID() const override
            {
                return "AttributeRegistry entry " + typeName_;
            }

        };

    }

}

#endif // MV_PASS_PASS_HPP_
