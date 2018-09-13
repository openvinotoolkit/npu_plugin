#ifndef ELEMENT_HPP_
#define ELEMENT_HPP_

#include <vector>
#include <string>
#include <map>
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/logger/log_sender.hpp"


namespace mv
{

    class Element : public Printable, public Jsonable, public LogSender
    {

        std::map<std::string, Attribute> attrs_;

    protected:

        std::string name_;

        virtual std::string attrsToString_() const;

    public:

        Element(const std::string &name);
        Element(const Element &other);
        Element(const std::string& name, const mv::json::Value& content);
        Element& operator=(const Element &other);
        virtual ~Element();

        const std::string &getName() const;
        void setName(const std::string& name);
        
        bool hasAttr(const std::string &name) const;
        std::size_t attrsCount() const;
        std::vector<std::string> attrsKeys() const;
        void clear();

        virtual std::string toString() const override;
        json::Value toJSON() const override;
        virtual std::string getLogID() const override;

        void set(const std::string& name, const Attribute& attr);

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
        void set(const std::string& name, const AttrType& value)
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
                attrs_.emplace(name, value);
            else
                attrs_[name] = value;
        }

        template <class AttrType>
        const AttrType& get(const std::string &name) const
        {
            if (!hasAttr(name))
                throw ArgumentError(*this, "attribute identifer", name,  "Undefined identifier");
            return attrs_.at(name).get<AttrType>();
        }

    };

}


#endif // COMPUTATION_ELEMENT_HPP_
