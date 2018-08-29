#ifndef COMPUTATION_ELEMENT_HPP_
#define COMPUTATION_ELEMENT_HPP_

#include <vector>
#include <string>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class ComputationElement : public Printable, public Jsonable, public LogSender
    {

    public:

        struct ElementOrderComparator
        {

            bool operator()(const std::weak_ptr<ComputationElement> &lhs, const std::weak_ptr<ComputationElement> &rhs)
            {
                //return lhs.lock()->getName() < rhs.lock()->getName();
                return *lhs.lock() < *rhs.lock();
            }

            /*bool operator()(const allocator::owner_ptr<ComputationElement> &lhs, const allocator::owner_ptr<ComputationElement> &rhs)
            {
                //return lhs->getName() < rhs->getName();
                return *lhs < *rhs;
            }*/
        };

    protected:

        friend class OpModel;
        friend class DataModel;
        friend class ComputationGroup;
        friend class ComputationStage;

        static Attribute unknownAttr_;
        std::string name_;
        std::map<std::string, Attribute> attributes_;

        template <class T>
        bool addAttr(const std::string &name, AttrType attrType, const T &content)
        {

            Attribute attr(attrType, content);
            return addAttr(name, attr);

        }

        bool addAttr(const std::string &name, const Attribute &attr);
        virtual std::string getLogID_() const override;

    public:

        ComputationElement(json::Value &value);
        ComputationElement(const std::string &name);
        ComputationElement(const ComputationElement &other);
        ComputationElement& operator=(const ComputationElement &other);
        virtual ~ComputationElement() = 0;
        const std::string &getName() const;
        void setName(const std::string& name);
        bool hasAttr(const std::string &name) const;
        Attribute& getAttr(const std::string &name);
        const Attribute& getAttr(const std::string &name) const;
        std::vector<std::string> getAttrKeys() const;
        AttrType getAttrType(const std::string &name) const;
        std::size_t attrsCount() const;
        bool removeAttr(const std::string &name);
        std::string toString() const;
        mv::json::Value virtual toJsonValue() const;
        virtual bool operator <(ComputationElement &other);
        virtual bool operator ==(const ComputationElement& other);

    };

}


#endif // COMPUTATION_ELEMENT_HPP_
