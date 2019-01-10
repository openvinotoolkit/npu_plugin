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
            Recurrent

        };

    private:

        const static unsigned jsonParserBufferLength_ = 128;

        /**
         * The map containing arguments for different passes.
         */
        std::map<std::string, Element> passesArgs_;

        /**
         * The map representing groups of passes. The first argument is the group id and the
         * second argument stores the list of groups/passes in that group. The recurrence of
         * a pass must be specified. XXX: do we need this kind of an enforcement?
         */
        std::map<std::string, std::pair<Recurrence, std::vector<std::string>> > groups_;

        /**
         * The root group containing the precise order of execution of groups.
         */
        std::vector<std::string> rootGroup_;

        static std::string toString(Recurrence rec);
        static Recurrence fromString(const std::string& str);

    public:

        CompilationDescriptor();
        CompilationDescriptor(const std::string& path);

        void load(const std::string& path);

        /**
         * Adds group to groups list. This just creates an entry for a group, without specifying
         * the passes in the group.
         */
        bool addGroup(const std::string& group);

        /**
         * Adds pass to groups list. If the group doesn't exist, create one. This method also
         * sets group recurrence -- "Singular" for a pass that executes only once; "Recurrent" for a
         * pass that recurs.
         */
        bool addPassToGroup(const std::string& pass, const std::string& group, const std::string& recurrence);

        /**
         * Explicitly set the root group. This is needed to specify the order in which the groups
         * will execute.
         */
        bool defineRootGroup(const std::vector<std::string>& rootGroup);

        /**
         * Add argument to a pass.
         */
        bool addArgToPass(const std::string& pass, const std::string& arg, const std::string& value);

    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_