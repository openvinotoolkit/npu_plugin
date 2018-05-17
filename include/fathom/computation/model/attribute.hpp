#ifndef ATTRIBUTE_HPP_
#define ATTRIBUTE_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/printable.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/constant.hpp"

template<mv::AttrType> struct AttrTypeToType { typedef void type; enum { value = false }; };
#define DEFINE_ENUMERATED_TYPE(TYPE, ATTRTYPE) template<> struct AttrTypeToType<ATTRTYPE> { typedef TYPE type; enum { value = true }; }

DEFINE_ENUMERATED_TYPE(mv::int_type, mv::AttrType::IntegerType);
DEFINE_ENUMERATED_TYPE(mv::unsigned_type, mv::AttrType::UnsingedType);
DEFINE_ENUMERATED_TYPE(mv::float_type, mv::AttrType::FloatType);
DEFINE_ENUMERATED_TYPE(mv::Shape, mv::AttrType::ShapeType);
DEFINE_ENUMERATED_TYPE(mv::byte_type, mv::AttrType::ByteType);
DEFINE_ENUMERATED_TYPE(mv::DType, mv::AttrType::DTypeType);
DEFINE_ENUMERATED_TYPE(mv::Order, mv::AttrType::OrderType);
DEFINE_ENUMERATED_TYPE(mv::ConstantTensor, mv::AttrType::TensorType);
DEFINE_ENUMERATED_TYPE(mv::string, mv::AttrType::StringType);

template<class T, class U>
struct is_same {
    enum { value = false };
};

template<class T>
struct is_same<T, T> {
    enum { value = true };
};

namespace mv
{

    class Attribute : public Printable
    {

    private:

        /*static byte_type nextTypeId();*/

        template <class T>
        static AttrType getTypeId()
        {
            //static byte_type result(nextTypeId());
            //return result;

            if ( AttrTypeToType<AttrType::ByteType>::value && is_same<T, AttrTypeToType<AttrType::ByteType>::type>::value)
                return AttrType::ByteType;
            
            if (AttrTypeToType<AttrType::DTypeType>::value  && is_same<T, AttrTypeToType<AttrType::DTypeType>::type>::value)
                return AttrType::DTypeType;

            if (AttrTypeToType<AttrType::FloatType>::value  && is_same<T, AttrTypeToType<AttrType::FloatType>::type>::value)
                return AttrType::FloatType;

            if (AttrTypeToType<AttrType::IntegerType>::value  && is_same<T, AttrTypeToType<AttrType::IntegerType>::type>::value)
                return AttrType::IntegerType;

            if (AttrTypeToType<AttrType::OrderType>::value  && is_same<T, AttrTypeToType<AttrType::OrderType>::type>::value)
                return AttrType::OrderType;
            
            if (AttrTypeToType<AttrType::ShapeType>::value  && is_same<T, AttrTypeToType<AttrType::ShapeType>::type>::value)
                return AttrType::ShapeType;

            if (AttrTypeToType<AttrType::TensorType>::value  && is_same<T, AttrTypeToType<AttrType::TensorType>::type>::value)
                return AttrType::TensorType;

            if (AttrTypeToType<AttrType::UnsingedType>::value  && is_same<T, AttrTypeToType<AttrType::UnsingedType>::type>::value)
                return AttrType::UnsingedType;
            
            if (AttrTypeToType<AttrType::StringType>::value  && is_same<T, AttrTypeToType<AttrType::StringType>::type>::value)
                return AttrType::StringType;

            return AttrType::UnknownType;

        }
        
        struct GenericAttributeContent
        {

            //byte_type typeId_;
            AttrType typeId_;

            GenericAttributeContent(AttrType typeId) : typeId_(typeId) {}
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
        T getContent() const
        {

            assert(getTypeId<T>() == content_->typeId_ && "Attribute type mismatch");
            return content_.cast_pointer<AttributeContent<T>>()->getContent();
        
        }

    };

}

#endif // ATTRIBUTE_HPP_