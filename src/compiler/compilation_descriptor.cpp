#include "include/mcm/compiler/compilation_descriptor.hpp"
#include "include/mcm/pass/pass_registry.hpp"

mv::CompilationDescriptor::CompilationDescriptor() :
Element("CompilationDescriptor")
{
}

mv::CompilationDescriptor::CompilationDescriptor(const std::string& path) :
Element("CompilationDescriptor")
{
    // Call load() to parse the json file passed in
}

bool mv::CompilationDescriptor::validPass(const std::string& passStr)
{
    bool pass_entry = mv::pass::PassRegistry::instance().find(passStr);

    if (!pass_entry) {
        return false;
    }

    return true;
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
    if (!hasAttr(group)) {
        throw AttributeError(*this, "Trying to add pass to a non-existent group (" + group + ")");
    }

    Element& g_elem = get<Element>(group);

    if (g_elem.hasAttr(recurrence)) {
        std::vector<std::string> &rec_v = g_elem.get<std::vector<std::string>>(recurrence);
        rec_v.push_back(elem);
    }
    else {
        std::vector<std::string> rec_v = { elem };
        g_elem.set<std::vector<std::string>>(recurrence, rec_v);
    }

    if (isGroup) {
        if (!hasAttr(elem)) {
            addGroup(elem);
        }
    }
    else {
        if (!validPass(elem)) {
            throw RuntimeError(*this, "Trying to pass (" + elem + "), not registered in the pass registry");
        }
    }
}

void mv::CompilationDescriptor::serializePassListInGroup(const std::string& group, std::vector<std::string> &serializedPasses)
{
    if (!hasAttr(group)) {
        throw AttributeError(*this, "Trying to serialize passes in a non-existent group (" + group + ")");
    }

    Element &elem = get<mv::Element>(group);

    std::vector<std::string> recurrentPasses;
    if (elem.hasAttr("Recurrent")) {
        std::vector<std::string> &recurrent_group = elem.get<std::vector<std::string>>("Recurrent");
        for (auto g: recurrent_group) {
            if (hasAttr(g)) {
                serializePassListInGroup(g, recurrentPasses);
            }
            else {
                recurrentPasses.push_back(g);
            }
        }
    }

    if (elem.hasAttr("Singular")) {
        std::vector<std::string> &singular_group = elem.get<std::vector<std::string>>("Singular");
        for (auto g: singular_group) {
            if (hasAttr(g)) {
                serializePassListInGroup(g, serializedPasses);
            }
            else {
                serializedPasses.push_back(g);
            }
            if (!recurrentPasses.empty()) {
                serializedPasses.insert(serializedPasses.end(), recurrentPasses.begin(), recurrentPasses.end());
            }
        }
    }
}

std::vector<std::string> mv::CompilationDescriptor::serializePassList()
{

    if (!hasAttr("root")) {
        throw RuntimeError(*this, "Unable to find root group, cannot serialize passes in the descriptor.");
    }

    std::vector<std::string> serializedPasses;

    serializePassListInGroup("root", serializedPasses);

    return serializedPasses;

}

void mv::CompilationDescriptor::printGroups(const std::string &groupStr)
{

    if (!hasAttr(groupStr)) {
        std::cout << "group(" << groupStr << ") doesn't exist" << std::endl;
    }

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
