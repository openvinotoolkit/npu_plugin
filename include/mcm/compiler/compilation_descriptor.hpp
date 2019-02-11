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
        void addElemAttribute(const std::string& elem, bool isGroup);
        bool isGroup(const std::string& elem);
        std::string profile_;

        /**
         * Adds a pass attribute to the compilation descriptor. The main reason we need this is to store arguments that
         * may accompany passes.
         */
        void addPass(const std::string& pass);
        std::string getElemString(const std::string &elem) const;

    public:

        CompilationDescriptor(const std::string& profile);
        CompilationDescriptor(const std::string& path, const std::string& profile);

        /**
         * Populate the compilation descriptor from a json file. Not implemented yet.
         */

        void load(const std::string& path);

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
         * Set argument to a pass.
         */
        void setPassArg(const std::string& pass, const std::string& arg, const mv::Attribute& value);

        /**
         * Get argument for a pass.
         */
        mv::Attribute getPassArg(const std::string& pass, const std::string& arg);

        /**
         * Validate a pass passed in.
         */
        bool validPass(const std::string& passStr);

        /**
         * Unfold groups into passes list.
         */
        std::vector<std::string> serializePassList();

        std::string toString() const override;

        std::string groupToString(const std::string &groupStr) const;
    };

}

#endif // MV_COMPILATION_DESCRIPTOR_HPP_