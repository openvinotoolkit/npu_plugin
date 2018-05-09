#ifndef ATTRIBUTE_HPP_
#define ATTRIBUTE_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/printable.hpp"

namespace mv
{

    class Attribute : public Printable
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

        static allocator allocator_;
        allocator::owner_ptr<GenericAttributeContent> content_;
        AttrType attrType_;

    public:

        

        Attribute();
        ~Attribute();
        AttrType getType() const;
        string toString() const;

        template <class T>
        Attribute(AttrType attrType, const T &content) : 
        content_(allocator_.make_owner<AttributeContent<T>>(content)),
        attrType_(attrType)
        {

        }

        template <class T>
        const T& getContent() const
        {
            assert(getTypeId<T>() == content_->typeId_ && "Attribute type mismatch");
            return cast_pointer<AttributeContent<T>>(content_)->getContent();
        }

    };

}

#endif // ATTRIBUTE_HPP_