#ifndef MV_COMPILATION_DESCRIPTOR_HPP_
#define MV_COMPILATION_DESCRIPTOR_HPP_

#include "include/mcm/base/element.hpp"

namespace mv
{

    class CompilationDescriptor : public Element
    {

    private:

        const static unsigned jsonParserBufferLength_ = 128;

        void serializePassListInGroup(const std::string& group, std::vector<std::string> &serializedPasses);

    public:

        CompilationDescriptor();
        CompilationDescriptor(const std::string& path);

        void load(const std::string& path);

        /**
         * Adds group to groups list. This just creates an entry for a group, without specifying
         * the passes in the group.
         */
        void addGroup(const std::string& group);

        /**
         * Adds some element to groups list. The element could be a group or a pass.
         * If the group doesn't exist, create one. This method also sets
         * recurrence -- "Singular" for a pass that executes only once; "Recurrent" for a
         * pass that recurs.
         */
        void addToGroup(const std::string& group, const std::string& elem, const std::string& recurrence, bool isGroup);

        /**
         * Add argument to a pass.
         */
        void addArgToPass(const std::string& pass, const std::string& arg, const std::string& value);

        /**
         * Validate a pass passed in.
         */
        bool validPass(const std::string& passStr);

        /**
         * Unfold groups into passes list.
         */
        std::vector<std::string> serializePassList();

        void printGroups(const std::string &groupStr);
    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_