#ifndef COMPUTATION_ELEMENT_HPP_
#define COMPUTATION_ELEMENT_HPP_

#include <string>
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/computation/logger/printable.hpp"

namespace mv
{

    class ComputationElement : public Printable
    {

        class Attribute
        {

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

        public:

            template <class T>
            Attribute(const T &content) : content_(allocator_.make_owner<AttributeContent<T>>(content)) {}
            Attribute() {}
            ~Attribute() {}

            template <class T>
            const T& getContent() const
            {
                assert(getTypeId<T>() == content_->typeId_ && "Attribute type mismatch");
                return cast_pointer<AttributeContent<T>>(content_)->getContent();
            }

        };

        static allocator allocator_;
        const Logger &logger_;

    protected:

        std::string name_;
        allocator::map<std::string, Attribute> attributes_;   

    public:

        ComputationElement(const Logger &logger, const std::string &name);
        virtual ~ComputationElement() = 0;
        const std::string &getName() const;

        template <class T>
        void addAttr(const std::string &name, const T &content)
        {
            logger_.log(Logger::MessageInfo, "Element '" + name_ + "' - adding attribute '" + name + "': " + toString(content));
            attributes_[name] = content;
        }

        template <class T>
        const T& getAttr(const std::string &name)
        {
            return attributes_[name].getContent<T>();
        }

    };

}


#endif // COMPUTATION_ELEMENT_HPP_