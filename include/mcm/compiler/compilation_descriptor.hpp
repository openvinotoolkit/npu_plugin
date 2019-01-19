#ifndef MV_COMPILATION_DESCRIPTOR_HPP_
#define MV_COMPILATION_DESCRIPTOR_HPP_

#include "include/mcm/base/element.hpp"

namespace mv
{

    class CompilationDescriptor : public Element
    {

    public:

        enum class RecurrenceType
        {

            Singular,
            Recurrent

        };

    private:

        const static unsigned jsonParserBufferLength_ = 128;

        static std::map<RecurrenceType, std::string> recTypeString_;

        /**
         * The map containing arguments for different passes.
         */
        //std::map<std::string, Element> passesArgs_;

        static std::string toString(const RecurrenceType& rec);
        static RecurrenceType fromString(const std::string& str);

    public:

        CompilationDescriptor();
        CompilationDescriptor(const std::string& path);

        void load(const std::string& path);

        /**
         * Adds group to groups list. This just creates an entry for a group, without specifying
         * the passes in the group.
         */
        void addGroup(const std::string& group);

        // TODO: Can the two functions below be combined?
        /**
         * Adds pass to groups list. If the group doesn't exist, create one. This method also
         * sets group recurrence -- "Singular" for a pass that executes only once; "Recurrent" for a
         * pass that recurs.
         */
        void addPassToGroup(const std::string& pass, const std::string& group, const std::string& recurrence);

        /**
         * Adds pass to groups list. If the group doesn't exist, create one. This method also
         * sets group recurrence -- "Singular" for a pass that executes only once; "Recurrent" for a
         * pass that recurs.
         */
        void addGroupToGroup(const std::string& group, const std::string& containerGroup, const std::string& recurrence);

        /**
         * Explicitly set the root group. This is needed to specify the order in which the groups
         * will execute. The root group is expected to be passed in as a vector of pairs of strings.
         * The first element in the pair specifies the recurrence, while the second element specifies
         * the group.
         */
        void defineRootGroup(const std::map<std::string, std::vector<std::string>>& groupList);

        /**
         * Add argument to a pass.
         */
        void addArgToPass(const std::string& pass, const std::string& arg, const std::string& value);

        /**
         * Unfold groups into passes list.
         */
        //std::vector<std::string> unfoldPasses();
        void unfoldPasses();

        void printGroups(const std::string &groupStr);

        //void printGroupElement(const std::map<RecurrenceType, std::vector<std::string>>& group);

    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_