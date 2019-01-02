#ifndef MV_BASE_ATTRIBUTE_HPP_
#define MV_BASE_ATTRIBUTE_HPP_

#include <type_traits>
#include <utility>
#include <typeinfo>
#include <typeindex>
#include <string>
#include <functional>
#include <initializer_list>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/base/binarizable.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{
    class Attribute : public Printable, public Jsonable, public LogSender, public Binarizable
    {
        
        struct AbstractObject
        {

            virtual ~AbstractObject()
            {

            }

            virtual AbstractObject* clone() const = 0;
            virtual std::string getTypeName() const = 0;
            virtual std::type_index getTypeID() const = 0;

        };

        template <class ValueType>
        struct Object : public AbstractObject
        {

            ValueType content_;
            const std::type_index typeID_;
            const std::string typeName_;

            template <class InitType>
            Object(InitType&& content) :
            content_(std::forward<InitType>(content)),
            typeID_(typeid(ValueType)),
            typeName_(attr::AttributeRegistry::getTypeName(typeID_))
            {

            }

            virtual ~Object()
            {

            }

            AbstractObject* clone() const override
            {
                return new Object(content_);
            }

            std::string getTypeName() const override
            {
                return typeName_;
            }

            std::type_index getTypeID() const override
            {
                return typeID_;
            }


        };

        AbstractObject* ptr_;
        std::set<std::string> traits_;
        std::function<Attribute(const mv::json::Value&)> fromJSONFunc_;

        AbstractObject* clone_() const
        {
            if (ptr_)
                return ptr_->clone();
            return nullptr;
        }

    public:

        template <class ValueType>
        Attribute(const ValueType& val, std::initializer_list<std::string> traits = {}) : 
        //ptr_(new Object<typename std::decay<ValueType>::type>(std::forward<ValueType>(val)))
        ptr_(new Object<typename std::decay<ValueType>::type>(val))
        {
            for (auto it = traits.begin(); it != traits.end(); ++it)
                addTrait(*it);
        }

        Attribute(const json::Object &val) :
        Attribute
        (
            [this, val]()->Attribute
            {
                if (!val.hasKey("attrType"))
                    throw AttributeError(*this, "Invalid JSON object passed for construction - missing field \"attrType\"");

                if (val["attrType"].valueType() != json::JSONType::String)
                    throw AttributeError(*this, "Invalid JSON object passed for construction - field \"attrType\" has a type of "
                        + json::Value::typeName(val["attrType"].valueType()) + " (should be string)");

                if (!attr::AttributeRegistry::checkType(val["attrType"].get<std::string>()))
                    throw AttributeError(*this, "Invalid JSON object passed for construction - field \"attrType\""
                        "specifies an unregistered type " + val["attrType"].get<std::string>());

                if (!val.hasKey("content"))
                    throw AttributeError(*this, "Invalid JSON object passed for construction - missing field \"content\"");

                if (val.hasKey("traits"))
                {
                    if (val["traits"].valueType() != json::JSONType::Array)
                        throw AttributeError(*this, "Invalid JSON object passed for construction - field \"traits\" has a type of "
                            + json::Value::typeName(val["traits"].valueType()) + " (should be array)");
                    
                    for (std::size_t i = 0; i < val["traits"].size(); ++i)
                    {
                        if (val["traits"][i].valueType() != json::JSONType::String)
                            throw AttributeError(*this, "Invalid JSON object passed for construction - field \"traits\" contains "
                            " an invalid value of type " + json::Value::typeName(val["traits"][i].valueType()) + " (should be string)");
                        traits_.insert(val["traits"][i].get<std::string>());
                    }
                }

                return attr::AttributeRegistry::getFromJSONFunc(val["attrType"].get<std::string>())(val["content"]);
            }()
        )
        {

        }

        Attribute(const json::Value &val) :
        Attribute
        (
            [this, val]()->const json::Object&
            {
                if (val.valueType() != json::JSONType::Object)
                throw AttributeError(*this, "Invalid JSON value passed for construction - value has a type of " +
                    json::Value::typeName(val.valueType()) + " (should be json::Object)");

                return val.get<json::Object>();
            }()
        )
        {

        }

        Attribute() : ptr_(nullptr)
        {

        }

        Attribute(Attribute& other) :
        ptr_(other.clone_()),
        traits_(other.traits_)
        {

        }

        Attribute(Attribute&& other) : 
        ptr_(other.ptr_),
        traits_(other.traits_)
        {
            other.ptr_ = nullptr;
            other.traits_.clear();
        }

        Attribute(const Attribute& other) :
        ptr_(other.clone_()),
        traits_(other.traits_)
        {

        }

        Attribute(const Attribute&& other) :
        ptr_(other.clone_()),
        traits_(other.traits_)
        {

        }

        ~Attribute()
        {
            if (ptr_)
                delete ptr_;
        }

        template <class ValueType>
        bool is() const
        {
            return dynamic_cast<Object<typename std::decay<ValueType>::type>*>(ptr_);
        }

        template <class ValueType>
        typename std::decay<ValueType>::type& get()
        {
            return const_cast<typename std::decay<ValueType>::type&>(static_cast<const Attribute*>(this)->get<ValueType>());
        }

        template <class ValueType>
        const typename std::decay<ValueType>::type& get() const
        {
            auto contentPtr_ = dynamic_cast<Object<typename std::decay<ValueType>::type>*>(ptr_);
            if (!contentPtr_)
                throw AttributeError(*this, "Attempt of obtaining the value of attribute of type " +
                    ptr_->getTypeName() + " as " + typeid(ValueType).name());

            return contentPtr_->content_;
        }

        template <class ValueType>
        operator ValueType()
        {
            return get<typename std::decay<ValueType>::type>();
        }

        Attribute& operator=(const Attribute& other)
        {

            if (ptr_ == other.ptr_)
                return *this;

            auto tPtr_ = ptr_;
            ptr_ = other.clone_();

            if (tPtr_)
                delete tPtr_;

            traits_ = other.traits_;
            return *this;
        }

        Attribute& operator=(Attribute&& other)
        {

            if (ptr_ == other.ptr_)
                return *this;

            std::swap(ptr_, other.ptr_);
            traits_ = other.traits_;
            return *this;

        }

        bool operator<(const Attribute& other) const
        {
            return ptr_ < other.ptr_;
        }

        bool operator==(const Attribute& other) const
        {
            return toJSON() == other.toJSON();
        }

        json::Value toJSON() const override
        {
            if (!ptr_)
                throw AttributeError(*this, "Uninitialized (null) attribute dereference called for to-JSON conversion");
            json::Object result;
            result["attrType"] = ptr_->getTypeName();
            result["content"] = attr::AttributeRegistry::getToJSONFunc(ptr_->getTypeID())(*this);
            if (!traits_.empty())
            {   
                result["traits"] = json::Array(traits_.size());
                std::size_t i = 0;
                for (auto it = traits_.begin(); it != traits_.end(); ++it)
                {
                    result["traits"][i] = *it;
                    ++i;
                }
            }
            
            return result;
        }

        std::string toString() const override
        {
            if (!ptr_)
                throw AttributeError(*this, "Uninitialized (null) attribute dereference called for to-string conversion");
            return attr::AttributeRegistry::getToStringFunc(ptr_->getTypeID())(*this);
        }

        std::string toLongString() const
        {
            return attr::AttributeRegistry::getToStringFunc(ptr_->getTypeID(), true)(*this);
        }

        std::vector<uint8_t> toBinary() const override
        {
            if (!ptr_)
                throw AttributeError(*this, "Uninitialized (null) attribute dereference called for to-binary conversion");
            return attr::AttributeRegistry::getToBinaryFunc(ptr_->getTypeID())(*this);
        }

        std::string getTypeName() const
        {
            if (!ptr_)
                throw AttributeError(*this, "Uninitialized (null) attribute dereference called for get attribute type name");
            return ptr_->getTypeName();
        }

        std::type_index getTypeID() const
        {
            if (!ptr_)
                throw AttributeError(*this, "Uninitialized (null) attribute dereference called for get attribute type ID");
            return ptr_->getTypeID();
        }

        bool valid() const
        {
            return ptr_ != nullptr;
        }

        bool hasTrait(const std::string& trait) const
        {
            return (traits_.find(trait) != traits_.end());
        }

        void addTrait(const std::string& trait)
        {
            if (!attr::AttributeRegistry::checkInstanceTrait(trait))
                throw AttributeError(*this, "Attempt of defining an illegal trait \"" + trait + "\"");
            traits_.insert(trait);
        }

        std::string getLogID() const override
        {
            if (!ptr_)
                return "Attribute:NULL";
            return "Attribute:" + getTypeName();
        }
    };

}

#endif // MV_BASE_ATTRIBUTE_HPP_
