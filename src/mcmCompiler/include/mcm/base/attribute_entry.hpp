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
            std::shared_ptr<GenericFunc> fromSimplifiedJSONFunc_;
            std::shared_ptr<GenericFunc> toSimplifiedJSONFunc_;
            std::shared_ptr<GenericFunc> toStringFunc_;
            std::shared_ptr<GenericFunc> toLongStringFunc_;
            std::shared_ptr<GenericFunc> toBinaryFunc_;
            std::set<std::string> typeTraits_;

        public:
            AttributeEntry(const std::type_index& typeID);

            AttributeEntry& setName(const std::string& typeName);
            AttributeEntry& setDescription(const std::string& description);
            AttributeEntry& setCheckFunc(const std::function<bool(const Attribute&, std::string&)>& f);
            AttributeEntry& setToJSONFunc(const std::function<mv::json::Value(const Attribute&)>& f);
            AttributeEntry& setFromJSONFunc(const std::function<Attribute(const mv::json::Value&)>& f);
            AttributeEntry& setToSimplifiedJSONFunc(const std::function<mv::json::Value(const Attribute&)>& f);
            AttributeEntry& setFromSimplifiedJSONFunc(const std::function<Attribute(const mv::json::Value&)>& f);
            AttributeEntry& setToStringFunc(const std::function<std::string(const Attribute&)>& f);
            AttributeEntry& setToBinaryFunc(const std::function<std::vector<uint8_t>(const Attribute&)>& f);
            AttributeEntry& setToLongStringFunc(const std::function<std::string(const Attribute&)>& f);
            AttributeEntry& setTypeTrait(const std::string& trait);

            const std::string& getTypeName() const;
            std::type_index getTypeID() const;
            const std::string& getDescription() const;
            const std::function<bool(const Attribute&, std::string&)>& getCheckFunc();
            const std::function<mv::json::Value(const Attribute&)>& getToJSONFunc();
            const std::function<Attribute(const mv::json::Value&)>& getFromJSONFunc();
            const std::function<mv::json::Value(const Attribute&)>& getToSimplifiedJSONFunc();
            const std::function<Attribute(const mv::json::Value&)>& getFromSimplifiedJSONFunc();
            const std::function<std::string(const Attribute&)>& getToStringFunc();
            const std::function<std::vector<uint8_t>(const Attribute&)>& getToBinaryFunc();
            const std::function<std::string(const Attribute&)>& getToLongStringFunc();
            bool hasTypeTrait(const std::string& trait);
            bool hasCheckFunc() const;

            std::string getLogID() const override;
        };

    }

}

#endif // MV_PASS_PASS_HPP_
