#ifndef MV_COMPILATION_DESCRIPTOR_HPP_
#define MV_COMPILATION_DESCRIPTOR_HPP_

#include "include/mcm/base/element.hpp"

namespace mv
{

    class CompilationDescriptor : public Element
    {

    private:

        const static unsigned jsonParserBufferLength_ = 128;

        // Maintain a list of groups in order to be able to delete groups        
        std::vector<std::string> groups_;

        void serializePassListInGroup(const std::string& group, std::vector<Element>& serializedPasses);
        bool isGroup(const std::string& elem);
        std::string profile_;

    public:

        CompilationDescriptor(const std::string& profile = "");
        CompilationDescriptor(const json::Object&, const std::string& profile = "");

        /**
         * Populate the compilation descriptor from a json file. Not implemented yet.
         */

        static json::Object load(const std::string& filePath);

        /**
         * Adds group attribute to the compilation descriptor. This just creates an entry for a group, without specifying
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
         * Remove element from specific group and a specific recurrence in that group.
         */
        void remove(const std::string& group, const std::string& elem, const std::string& recurrence);

        /**
         * Remove element from all recurrences in a group (Singular and Recurrent). Remove all references
         * to said group from other groups as well.
         */
        void remove(const std::string& group, const std::string& pass);

        /**
         * Remove group. Remove all references to said group from other groups as well.
         */
        void remove(const std::string& group);

        /**
         * Remove all groups and passes.
         */
        void clear();

        /**
         * Get number of groups, including root group.
         */
        size_t getNumGroups();

        /**
         * Check whether root group is defined.
         */
        bool rootGroupPresent();

        /**
         * Set argument to a pass in all groups.
         */
        void setPassArg(const std::string& pass, const std::string& arg, const mv::Attribute& value);
        
        /**
         * Set argument to a pass in a specific group only.
         */
        void setPassArg(const std::string& group, const std::string& pass, const std::string& arg, const mv::Attribute& value);

        /**
         * Get argument for a pass.
         */
        mv::Attribute getPassArg(const std::string& group, const std::string& recType, const std::string& pass, const std::string& arg);

        /**
         * Validate a pass passed in.
         */
        bool validPass(const std::string& passStr);

        /**
         * Unfold groups into passes list.
         */
        std::vector<mv::Element> serializePassList();

    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_
