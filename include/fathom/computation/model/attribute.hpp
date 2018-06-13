#ifndef ATTRIBUTE_HPP_
#define ATTRIBUTE_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/logger/printable.hpp"

template<mv::AttrType> struct AttrTypeToType { typedef void type; enum { value = false }; };
#define DEFINE_ENUMERATED_TYPE(TYPE, ATTRTYPE) template<> struct AttrTypeToType<ATTRTYPE> { typedef TYPE type; enum { value = true }; }

DEFINE_ENUMERATED_TYPE(mv::int_type, mv::AttrType::IntegerType);
DEFINE_ENUMERATED_TYPE(mv::unsigned_type, mv::AttrType::UnsingedType);
DEFINE_ENUMERATED_TYPE(mv::float_type, mv::AttrType::FloatType);
DEFINE_ENUMERATED_TYPE(mv::Shape, mv::AttrType::ShapeType);
DEFINE_ENUMERATED_TYPE(mv::byte_type, mv::AttrType::ByteType);
DEFINE_ENUMERATED_TYPE(mv::DType, mv::AttrType::DTypeType);
DEFINE_ENUMERATED_TYPE(mv::Order, mv::AttrType::OrderType);
DEFINE_ENUMERATED_TYPE(mv::string, mv::AttrType::StringType);
DEFINE_ENUMERATED_TYPE(bool, mv::AttrType::BoolType);
DEFINE_ENUMERATED_TYPE(mv::OpType, mv::AttrType::OpTypeType);
DEFINE_ENUMERATED_TYPE(mv::FloatVector2D, mv::AttrType::FloatVec2DType);
DEFINE_ENUMERATED_TYPE(mv::FloatVector3D, mv::AttrType::FloatVec3DType);
DEFINE_ENUMERATED_TYPE(mv::FloatVector4D, mv::AttrType::FloatVec4DType);
DEFINE_ENUMERATED_TYPE(mv::IntVector2D, mv::AttrType::IntVec2DType);
DEFINE_ENUMERATED_TYPE(mv::IntVector3D, mv::AttrType::IntVec3DType);
DEFINE_ENUMERATED_TYPE(mv::IntVector4D, mv::AttrType::IntVec4DType);
DEFINE_ENUMERATED_TYPE(mv::UnsignedVector2D, mv::AttrType::UnsignedVec2DType);
DEFINE_ENUMERATED_TYPE(mv::UnsignedVector3D, mv::AttrType::UnsignedVec3DType);
DEFINE_ENUMERATED_TYPE(mv::UnsignedVector4D, mv::AttrType::UnsignedVec4DType);


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

        template <class T>
        static AttrType getTypeId()
        {

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

            if (AttrTypeToType<AttrType::UnsingedType>::value  && is_same<T, AttrTypeToType<AttrType::UnsingedType>::type>::value)
                return AttrType::UnsingedType;
            
            if (AttrTypeToType<AttrType::StringType>::value  && is_same<T, AttrTypeToType<AttrType::StringType>::type>::value)
                return AttrType::StringType;

            if (AttrTypeToType<AttrType::BoolType>::value  && is_same<T, AttrTypeToType<AttrType::BoolType>::type>::value)
                return AttrType::BoolType;

            if (AttrTypeToType<AttrType::OpTypeType>::value  && is_same<T, AttrTypeToType<AttrType::OpTypeType>::type>::value)
                return AttrType::OpTypeType;

            if (AttrTypeToType<AttrType::FloatVec2DType>::value  && is_same<T, AttrTypeToType<AttrType::FloatVec2DType>::type>::value)
                return AttrType::FloatVec2DType;
            
            if (AttrTypeToType<AttrType::FloatVec3DType>::value  && is_same<T, AttrTypeToType<AttrType::FloatVec3DType>::type>::value)
                return AttrType::FloatVec3DType;

            if (AttrTypeToType<AttrType::FloatVec4DType>::value  && is_same<T, AttrTypeToType<AttrType::FloatVec4DType>::type>::value)
                return AttrType::FloatVec4DType;

            if (AttrTypeToType<AttrType::IntVec2DType>::value  && is_same<T, AttrTypeToType<AttrType::IntVec2DType>::type>::value)
                return AttrType::IntVec2DType;
            
            if (AttrTypeToType<AttrType::IntVec3DType>::value  && is_same<T, AttrTypeToType<AttrType::IntVec3DType>::type>::value)
                return AttrType::IntVec3DType;

            if (AttrTypeToType<AttrType::IntVec4DType>::value  && is_same<T, AttrTypeToType<AttrType::IntVec4DType>::type>::value)
                return AttrType::IntVec4DType;

            if (AttrTypeToType<AttrType::UnsignedVec2DType>::value  && is_same<T, AttrTypeToType<AttrType::UnsignedVec2DType>::type>::value)
                return AttrType::UnsignedVec2DType;
            
            if (AttrTypeToType<AttrType::UnsignedVec3DType>::value  && is_same<T, AttrTypeToType<AttrType::UnsignedVec3DType>::type>::value)
                return AttrType::UnsignedVec3DType;

            if (AttrTypeToType<AttrType::UnsignedVec4DType>::value  && is_same<T, AttrTypeToType<AttrType::UnsignedVec4DType>::type>::value)
                return AttrType::UnsignedVec4DType;

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
            void setContent(const T &content) { content_ = content;}

        };

        static allocator allocator_;
        allocator::owner_ptr<GenericAttributeContent> content_;
        AttrType attrType_;

    public:

        Attribute();
        ~Attribute();
        AttrType getType() const;
        string toString() const;
        string getContentStr() const;

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

        template <class T>
        void setContent(const T &content)
        {

            assert(getTypeId<T>() == content_->typeId_ && "Attribute type mismatch");
            return content_.cast_pointer<AttributeContent<T>>()->setContent(content);
        
        }

    };

}

#endif // ATTRIBUTE_HPP_