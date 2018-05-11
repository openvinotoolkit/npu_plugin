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

    protected:

        friend class OpModel;

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
        virtual ~ComputationElement() = 0;
        const string &getName() const;
        Attribute getAttr(const string &name);
        AttrType getAttrType(const string &name);
        unsigned_type attrsCount() const;
        string toString() const;

    };

}


#endif // COMPUTATION_ELEMENT_HPP_