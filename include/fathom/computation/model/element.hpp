#ifndef COMPUTATION_ELEMENT_HPP_
#define COMPUTATION_ELEMENT_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/computation/logger/printable.hpp"

namespace mv
{

    class ComputationElement : public Printable
    {

    public:

        enum class AttrType
        {
            
            UnknownType,
            ByteType,
            UnsingedType,
            IntegerType,
            FloatType,
            TensorType,
            DTypeType,
            OrderType,
            ShapeType

        };

    protected:

        class Attribute
        {

        private:

            static byte_type nextTypeId();

            template <class T>
            static byte_type getTypeId()
            {
                static byte_type result(nextTypeId());
                return result;
            }
            
            struct GenericAttributeContent
            {

                byte_type typeId_;

                GenericAttributeContent(const byte_type typeId) : typeId_(typeId) {}
                virtual ~GenericAttributeContent() {}

            };

            template<class T>
            class AttributeContent : public GenericAttributeContent
            {
                
                T content_;

            public:

                AttributeContent(const T &content) : GenericAttributeContent(getTypeId<T>()), content_(content) {};
                T& getContent() { return content_; }

            };

            allocator::owner_ptr<GenericAttributeContent> content_;
            AttrType attrType_;

        public:

            template <class T>
            Attribute(AttrType attrType, const T &content) : 
            content_(allocator_.make_owner<AttributeContent<T>>(content)),
            attrType_(attrType)
            {

            }

            Attribute() :
            attrType_(AttrType::UnknownType)
            {

            }

            ~Attribute() {}

            template <class T>
            const T& getContent() const
            {
                assert(getTypeId<T>() == content_->typeId_ && "Attribute type mismatch");
                return cast_pointer<AttributeContent<T>>(content_)->getContent();
            }

            AttrType getType() const
            {
                return attrType_;
            }

        };

        static allocator allocator_;

        const Logger &logger_;

        string name_;
        allocator::map<string, Attribute> attributes_;

    public:

        ComputationElement(const Logger &logger, const string &name);
        virtual ~ComputationElement() = 0;
        const string &getName() const;

        template <class T>
        bool addAttr(const string &name, AttrType attrType, const T &content)
        {
            if (attributes_.find(name) == attributes_.end())
            {
                logger_.log(Logger::MessageType::MessageDebug, "Element '" + name_ + "' - adding attribute '" + name + "': " + Printable::toString(content));
                attributes_[name] = Attribute(attrType, content);
                return true;
            }
            else
            {
                logger_.log(Logger::MessageType::MessageError, "Element '" + name_ + "' - failed adding attribute of duplicated name '" + name + "'");
                return false;
            }
        }

        template <class T>
        Attribute getAttr(const string &name)
        {
            if (attributes_.find(name) != attributes_.end())
                return attributes_[name];
            else
                return Attribute();

        }

        string toString() const;

    };

}


#endif // COMPUTATION_ELEMENT_HPP_