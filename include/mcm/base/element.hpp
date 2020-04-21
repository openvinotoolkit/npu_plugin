#ifndef ELEMENT_HPP_
#define ELEMENT_HPP_

#include <vector>
#include <string>
#include <map>
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{

    class Element : public Printable, public Jsonable, public LogSender
    {

        std::map<std::string, Attribute> attrs_;

    protected:

        std::string name_;
        virtual std::string attrsToString_() const;
        void forceErase_(const std::string& name);
        const std::map<std::string, Attribute>& getAttrs_() const;

    public:

        Element(const std::string& name);
        Element(const char* name);
        Element(const Element& other);
        virtual std::map<std::string, Attribute> getAttrs(const std::vector<std::string>& forbiddenKeys = {}) const;
        virtual std::map<std::string, Attribute> attrsToCopy(const std::vector<std::string>& forbiddenKeys = {}) const;
        void setAttrs(const std::map<std::string, Attribute>& attrs);

        /**
         * @brief Construct a new Element object from an JSON::Object
         *
         * @param content Input JSON::Object, other JSON types are invalid
         * @param autoType Enable automatic type deduction. Allows to simplify the JSON input,
         * but restricts types of input Attributes to JSON types.
         */
        Element(const mv::json::Value& content, bool simplifiedTyping = false, std::string name = "");
        Element& operator=(const Element& other);
        bool operator<(const Element& other) const;
        bool operator==(const Element& other) const;
        virtual ~Element();

        const std::string &getName() const;
        void setName(const std::string& name);

        bool hasAttr(const std::string &name) const;
        std::size_t attrsCount() const;
        std::vector<std::string> attrsKeys() const;
        void clear();

        virtual std::string toString() const override;
        virtual json::Value toJSON() const override;
        virtual json::Value toJSON(bool simplifiedTyping) const;
        virtual std::string getLogID() const override;

        Attribute& get(const std::string& name);
        void set(const std::string& name, const Attribute& attr);
        void erase(const std::string& name);

/*template <class AttrType>
        void set(const std::string& name, AttrType&& value)
        {
            if (!attr::AttributeRegistry::checkType<AttrType>())
                throw AttributeError("Unable to define the attribute '" + name + "' of an undefined"
                    " type " + typeid(AttrType).name());
            Attribute newAttr = value;
            std::string errMsg;
            if (!attr::AttributeRegistry::checkValue<AttrType>(newAttr, errMsg))
                throw AttributeError("Unable to define the attribute '" + name + "' of type "
                    + newAttr.getTypeName() + " with an invalid value - " + errMsg);
            if (!hasAttr(name))
                attrs_.emplace(name, std::move(value));
            else
                attrs_[name] = std::move(value);
        }*/

        template <class AttrType>
        void set(const std::string& name, const AttrType& value, std::initializer_list<std::string> traits = {})
        {
            if (!attr::AttributeRegistry::checkType<AttrType>())
                throw ArgumentError(*this, "type", typeid(AttrType).name(), "Unregistered"
                    " type used for Attribute " + name + "initialization");

            Attribute newAttr = value;
            std::string errMsg;
            if (!attr::AttributeRegistry::checkValue<AttrType>(newAttr, errMsg))
                throw ArgumentError(*this, "attribute value", attr::AttributeRegistry::getToStringFunc(typeid(AttrType))(value),
                    "Invalid value used for initialization of Attribute " + name + " - " + errMsg);

            if (!hasAttr(name))
            {
                auto it = attrs_.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(value, traits));
                if (!it.second)
                    throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");

                log(Logger::MessageType::Debug, "Attribute '" + name + "' (" + it.first->second.getTypeName() +
                    ") set to " + it.first->second.toString());
            }
            else
            {
                if (attrs_[name].hasTrait("const"))
                    throw AttributeError(*this, "Attempt of modification of a const attribute " + name);
                attrs_[name] = value;

                log(Logger::MessageType::Debug, "Attribute '" + name + "' (" + attrs_[name].getTypeName() +
                    ") modified to " + attrs_[name].toString());
            }
        }

        template <class AttrType>
        const AttrType& get(const std::string &name) const
        {
            if (!hasAttr(name))
                throw ArgumentError(*this, "attribute identifer", name,  "Undefined identifier");
            return attrs_.at(name).get<AttrType>();
        }

        template <class AttrType>
        AttrType& get(const std::string &name)
        {
            if (!hasAttr(name))
                throw ArgumentError(*this, "attribute identifer", name,  "Undefined identifier");
            return attrs_.at(name).get<AttrType>();
        }

    };

}


#endif // COMPUTATION_ELEMENT_HPP_
