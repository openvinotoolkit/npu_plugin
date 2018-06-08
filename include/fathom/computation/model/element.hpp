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
                //return lhs.lock()->getName() < rhs.lock()->getName();
                return *lhs < *rhs;
            }

            /*bool operator()(const allocator::owner_ptr<ComputationElement> &lhs, const allocator::owner_ptr<ComputationElement> &rhs)
            {
                //return lhs->getName() < rhs->getName();
                return *lhs < *rhs;
            }*/
        };

    protected:

        friend class OpModel;
        friend class ComputationGroup;
        friend class ComputationStage;

        static allocator allocator_;
        static Attribute unknownAttr_;
        static Logger &logger_;
        string name_;
        map<string, Attribute> attributes_;

        template <class T>
        bool addAttr(const string &name, AttrType attrType, const T &content)
        {

            Attribute attr(attrType, content);
            return addAttr(name, attr);

        }

        bool addAttr(const string &name, const Attribute &attr);

    public:

        ComputationElement(const string &name);
        ComputationElement(const ComputationElement &other);
        ComputationElement& operator=(const ComputationElement &other);
        virtual ~ComputationElement() = 0;
        const string &getName() const;
        bool hasAttr(const string &name) const;
        Attribute& getAttr(const string &name);
        const Attribute& getAttr(const string &name) const;
        allocator::vector<string> getAttrKeys() const;
        AttrType getAttrType(const string &name) const;
        unsigned_type attrsCount() const;
        bool removeAttr(const string &name);
        string toString() const;
        virtual bool operator <(ComputationElement &other);

    };

}


#endif // COMPUTATION_ELEMENT_HPP_