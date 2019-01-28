#include "include/mcm/compiler/compilation_descriptor.hpp"
#include "include/mcm/pass/pass_registry.hpp"

mv::CompilationDescriptor::CompilationDescriptor(const std::string& profile) :
Element("CompilationDescriptor"),
profile_(profile)
{
}

mv::CompilationDescriptor::CompilationDescriptor(const std::string& path, const std::string& profile) :
Element("CompilationDescriptor"),
profile_(profile)
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

void mv::CompilationDescriptor::addElemAttribute(const std::string& elem, bool isGroup)
{
    if (!hasAttr(elem))
    {
        Element e = Element(elem);

        if (isGroup)
            e.set<bool>("isGroup", true);
        else
            e.set<bool>("isGroup", false);

        set<Element>(elem, e);
    }
}

void mv::CompilationDescriptor::addGroup(const std::string& group)
{
    addElemAttribute(group, true);
}

void mv::CompilationDescriptor::addPass(const std::string& pass)
{
    addElemAttribute(pass, false);
}

void mv::CompilationDescriptor::addToGroup(const std::string& group, const std::string& elem, const std::string& recurrence, bool elemIsGroup)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to add pass to a non-existent group (" + group + ")");

    if (!elemIsGroup && !validPass(elem))
        throw RuntimeError(*this, "Trying to add pass (" + elem + "), not registered in the pass registry");

    Element& g_elem = get<Element>(group);

    if (g_elem.hasAttr(recurrence))
    {
        std::vector<std::string> &rec_v = g_elem.get<std::vector<std::string>>(recurrence);
        rec_v.push_back(elem);
    }
    else
    {
        std::vector<std::string> rec_v = { elem };
        g_elem.set<std::vector<std::string>>(recurrence, rec_v);
    }

    if (!hasAttr(elem))
    {
        if (elemIsGroup)
            addGroup(elem);
        else
            addPass(elem);
    }
}

bool mv::CompilationDescriptor::isGroup(const std::string& elem)
{
    if (!hasAttr(elem))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Non-existent element (" + elem + ")");

    Element &e = get<Element>(elem);

    return e.get<bool>("isGroup");
}

void mv::CompilationDescriptor::setPassArg(const std::string& pass, const std::string& arg, const std::string& value)
{
    if (!hasAttr(pass))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to add arguments to a non-existent pass (" + pass + ")");

    Element& p = get<Element>(pass);
    p.set<std::string>(arg, value);
}

std::string mv::CompilationDescriptor::getPassArg(const std::string& pass, const std::string& arg)
{
    if (!hasAttr(pass))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to get arguments from a non-existent pass (" + pass + ")");

    Element& p = get<Element>(pass);

    return p.get<std::string>(arg);

}

void mv::CompilationDescriptor::serializePassListInGroup(const std::string& group, std::vector<std::string> &serializedPasses)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to serialize passes in a non-existent group (" + group + ")");

    Element &elem = get<mv::Element>(group);

    std::vector<std::string> recurrentPasses;
    if (elem.hasAttr("Recurrent"))
    {
        std::vector<std::string> &recurrent_group = elem.get<std::vector<std::string>>("Recurrent");
        for (auto g: recurrent_group)
        {
            if (isGroup(g))
                serializePassListInGroup(g, recurrentPasses);
            else
                recurrentPasses.push_back(g);
        }
    }

    if (elem.hasAttr("Singular"))
    {
        std::vector<std::string> &singular_group = elem.get<std::vector<std::string>>("Singular");
        for (auto g: singular_group)
        {
            if (isGroup(g))
                serializePassListInGroup(g, serializedPasses);
            else
                serializedPasses.push_back(g);

            if (!recurrentPasses.empty())
                serializedPasses.insert(serializedPasses.end(), recurrentPasses.begin(), recurrentPasses.end());
        }
    }
}

std::vector<std::string> mv::CompilationDescriptor::serializePassList()
{

    if (!hasAttr("root"))
        throw RuntimeError(*this, "Unable to find root group, cannot serialize passes in the descriptor.");

    std::vector<std::string> serializedPasses;

    serializePassListInGroup("root", serializedPasses);

    return serializedPasses;

}

std::string mv::CompilationDescriptor::getElemString(const std::string &elem) const
{
    std::string output;

    if (hasAttr(elem) && get<Element>(elem).get<bool>("isGroup"))
    {
        output += "    group: " + elem + "\n";
        output += groupToString(elem);
    }
    else
        output += "    pass : " + elem + "\n";

    return output;
}

std::string mv::CompilationDescriptor::groupToString(const std::string &groupStr) const
{
    if (!hasAttr(groupStr))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Invalid group passed in (" + groupStr + ")");

    const Element &group = get<Element>(groupStr);

    std::string output;
    output += "group: " + groupStr + "\n";

    if (group.hasAttr("Singular"))
    {
        const std::vector<std::string> &sing_group = group.get<std::vector<std::string>>("Singular");
        output += "  Singular: " + groupStr + "\n";
        for (auto g: sing_group)
            output += getElemString(g);
    }

    if (group.hasAttr("Recurrent"))
    {
        const std::vector<std::string> &rec_group = group.get<std::vector<std::string>>("Recurrent");
        output += "  Recurrent: " + groupStr + "\n";
        for (auto g: rec_group)
            output += getElemString(g);
    }

    return output;
}

std::string mv::CompilationDescriptor::toString() const
{
    std::string output;

    output += "CompilationDescriptor:\n";
    output += "profile: " + profile_ + "\n";

    if (hasAttr("root"))
        output += groupToString("root");

    return output;
}