#ifndef MV_COMPILATION_DESCRIPTOR_HPP_
#define MV_COMPILATION_DESCRIPTOR_HPP_

#include "include/mcm/base/element.hpp"

namespace mv
{

    class CompilationDescriptor
    {

    public:

        enum class Recurrence
        {

            Singular,
            GroupWise,
            PassWise

        };

    private:

        const static unsigned jsonParserBufferLength_ = 128;

        std::map<std::string, Element> passesArgs_;
        std::map<std::string, std::vector<std::string>> passesGroups_;
        std::vector<std::pair<std::string, Recurrence>> groupsOrder_;        

        static std::string toString(Recurrence rec);
        static Recurrence fromString(const std::string& str);

    public:

        CompilationDescriptor();
        CompilationDescriptor(const std::string& path);

        void load(const std::string& path);

    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_