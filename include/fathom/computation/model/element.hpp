#ifndef COMPUTATION_ELEMENT_HPP_
#define COMPUTATION_ELEMENT_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/computation/logger/printable.hpp"
#include "include/fathom/computation/model/attribute.hpp"

namespace mv
{

    class ComputationElement : public Printable
    {

    public:

        struct ElementOrderComparator
        {

            bool operator()(const allocator::access_ptr<ComputationElement> &lhs, const allocator::access_ptr<ComputationElement> &rhs)
            {
                return lhs.lock()->getName() < rhs.lock()->getName();
            }

            bool operator()(const allocator::owner_ptr<ComputationElement> &lhs, const allocator::owner_ptr<ComputationElement> &rhs)
            {
                return lhs->getName() < rhs->getName();
            }
        };

    protected:

        friend class OpModel;
        friend class ComputationGroup;

        static allocator allocator_;
        const Logger &logger_;
        string name_;
        allocator::map<string, Attribute> attributes_;

        template <class T>
        bool addAttr(const string &name, AttrType attrType, const T &content)
        {

            Attribute attr(attrType, content);
            return addAttr(name, attr);

        }

        bool addAttr(const string &name, const Attribute &attr);

    public:

        ComputationElement(const Logger &logger, const string &name);
        ComputationElement(const ComputationElement &other);
        ComputationElement& operator=(const ComputationElement &other);
        virtual ~ComputationElement() = 0;
        const string &getName() const;
        bool hasAttr(const string &name);
        Attribute getAttr(const string &name);
        vector<string> getAttrKeys() const;
        AttrType getAttrType(const string &name);
        unsigned_type attrsCount() const;
        string toString() const;
    
    };

}


#endif // COMPUTATION_ELEMENT_HPP_