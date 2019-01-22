#include "include/mcm/compiler/compilation_descriptor.hpp"

std::map<mv::CompilationDescriptor::RecurrenceType, std::string> mv::CompilationDescriptor::recTypeString_
{
    {RecurrenceType::Singular, "Singular"},
    {RecurrenceType::Recurrent, "Recurrent"}
};

mv::CompilationDescriptor::RecurrenceType mv::CompilationDescriptor::fromString(const std::string& str)
{
    for (auto type: mv::CompilationDescriptor::recTypeString_) {
        if (type.second == str) {
            return type.first;
        }
    }
}

std::string mv::CompilationDescriptor::toString(const mv::CompilationDescriptor::RecurrenceType &rec)
{
    std::string recTypeString;
    for (auto recType: mv::CompilationDescriptor::recTypeString_) {
        if (recType.first == rec) {
             recTypeString = recType.second;
        }
    }

    return recTypeString;
}

mv::CompilationDescriptor::CompilationDescriptor() :
Element("CompilationDescriptor")
{
}

mv::CompilationDescriptor::CompilationDescriptor(const std::string& path) :
Element("CompilationDescriptor")
{
    // Call load() to parse the json file passed in
}

void mv::CompilationDescriptor::addGroup(const std::string& group)
{
    if (!hasAttr(group)) {
        Element g = Element(group);
        g.set<bool>("isGroup", true);
        set<Element>(group, g);
    }
}

void mv::CompilationDescriptor::addToGroup(const std::string& group, const std::string& elem, const std::string& recurrence, bool isGroup)
{
    // TODO: verify that pass is a valid one
    if (hasAttr(group)) {
        Element& g_elem = get<Element>(group);

        if (g_elem.hasAttr(recurrence)) {
            std::vector<std::string> &rec_v = g_elem.get<std::vector<std::string>>(recurrence);
            rec_v.push_back(elem);
        }
        else {
            std::vector<std::string> rec_v = { elem };
            g_elem.set<std::vector<std::string>>(recurrence, rec_v);
        }

        if (isGroup && !hasAttr(elem)) {
            addGroup(elem);
        }
    }
    else {
        throw AttributeError(*this, "Trying to add pass to a non-existent group (" + group + ")");
    }
}

void mv::CompilationDescriptor::defineRootGroup(const std::map<std::string, std::vector<std::string>>& groupList)
{

    if (!hasAttr("root")) {
        addGroup("root");
    }

    for (auto listItem: groupList) {
        std::vector<std::string> group_vec = listItem.second;
        for (auto group: group_vec) {
            addPassToGroup(group, "root", listItem.first);
        }
    }

}

void mv::CompilationDescriptor::unfoldPasses()
{
    // 1) Get rootGroup
    // 2) walk through the rootGroup and start unfolding groups and passes.

    std::vector<std::string> passes;

    if (!hasAttr("root")) {
        throw RuntimeError(*this, "Unable to find root group, cannot serialize pass list");
    }

    Element &root = get<Element>("root");

    if (root.hasAttr("Recurrent")) {
        std::vector<std::string> &rec_group = root.get<std::vector<std::string>>("Recurrent");
        for (auto group: rec_group) {
            printGroups(group);
        }
    }

    if (root.hasAttr("Singular")) {
        std::vector<std::string> &sing_group = root.get<std::vector<std::string>>("Singular");
        for (auto group: sing_group) {
            printGroups(group);
        }
    }
}

void mv::CompilationDescriptor::printGroups(const std::string &groupStr)
{

    if (hasAttr(groupStr)) {
        Element &group = get<Element>(groupStr);

        if (group.hasAttr("Singular")) {
            std::vector<std::string> &sing_group = group.get<std::vector<std::string>>("Singular");
            std::cout << "group(" << groupStr << ") - singular:" << std::endl;
            for (auto g: sing_group) {
                if (hasAttr(g) && get<Element>(g).get<bool>("isGroup")) {
                    std::cout << " element " << g << " is a group" << std::endl;
                }
                else {
                    std::cout << " element " << g << " is a pass" << std::endl;
                }

            }
        }

        if (group.hasAttr("Recurrent")) {
            std::vector<std::string> &rec_group = group.get<std::vector<std::string>>("Recurrent");
            std::cout << "group(" << groupStr << ") - recurrent:" << std::endl;
            for (auto g: rec_group) {
                if (hasAttr(g) && get<Element>(g).get<bool>("isGroup")) {
                    std::cout << " element " << g << " is a group" << std::endl;
                }
                else {
                    std::cout << " element " << g << " is a pass" << std::endl;
                }
            }
        }
    }
    else {
        std::cout << "group(" << groupStr << ") doesn't exist" << std::endl;
    }

}
