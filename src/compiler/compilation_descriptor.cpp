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
    return mv::pass::PassRegistry::instance().find(passStr) ? true : false;
}

void mv::CompilationDescriptor::addGroup(const std::string& group)
{
    if (!hasAttr(group))
    {
        Element g = Element(group);
        g.set<bool>("isGroup", true);
        set<Element>(group, g);
    }

    auto it = std::find(groups_.begin(), groups_.end(), group);
    if (it == groups_.end())
        groups_.push_back(group);
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

    if (elemIsGroup)
    {
        if (!hasAttr(elem))
            addGroup(elem);
    }
    else
    {
        if (!g_elem.hasAttr(elem))
        {
            Element p = Element(elem);
            p.set<bool>("isGroup", false);
            g_elem.set<Element>(elem, p);
        }
    }
}

void mv::CompilationDescriptor::remove(const std::string& group, const std::string& elem, const std::string& recurrence)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to remove pass from a non-existent group (" + group + ")");
    
    Element& g_elem = get<Element>(group);

    if (!g_elem.hasAttr(recurrence))
        throw RuntimeError(*this, recurrence + " recurrence is not present in group " + group);

    // Remove element from list of passes.
    auto& rec_v = g_elem.get<std::vector<std::string>>(recurrence);
    
    auto it = std::find(rec_v.begin(), rec_v.end(), elem);
    if (it == rec_v.end())
        throw RuntimeError(*this, "Unable to find element (" + elem + ") in group (" + group + ")");
    
    rec_v.erase(it);

    // Remove pass attribute from the group (if elem happens to be a pass).
    if (g_elem.hasAttr(elem))
        g_elem.erase(elem);
}

void mv::CompilationDescriptor::remove(const std::string& group, const std::string& elem)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to remove element from a non-existent group (" + group + ")");

    Element& g_elem = get<Element>(group);

    if (g_elem.hasAttr("Singular"))
        remove(group, elem, "Singular");

    if (g_elem.hasAttr("Recurrent"))
        remove(group, elem, "Recurrent");
}

void mv::CompilationDescriptor::remove(const std::string& group)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to remove a non-existent group (" + group + ")");
    
    // Remove group attribute from descriptor.
    erase(group);

    // Remove entry from root group, if root group is present.
    if (group != "root" && hasAttr("root"))
    {
        auto& root = get<Element>("root");
        if (root.hasAttr(group))
            root.erase(group);
    }

    for (auto g: groups_)
    {
        if (hasAttr(g))
        {
            // Find group in g's recurrence lists, and remove it
            auto removeInRecurrenceGroup = [this, g, group](const std::string& recType) {
                auto& elem = get<Element>(g);
                if (elem.hasAttr(recType)) {
                    auto& recList = elem.get<std::vector<std::string>>(recType);
                    auto it = std::find(recList.begin(), recList.end(), group);
                    if (it != recList.end())
                        recList.erase(it);
                }
            };

            removeInRecurrenceGroup("Singular");
            removeInRecurrenceGroup("Recurrent");
        }
    }

    // Remove group from list of groups
    auto it = std::find(groups_.begin(), groups_.end(), group);
    if (it != groups_.end()) {
        groups_.erase(it);
    }
}

void mv::CompilationDescriptor::clear()
{
    for (auto group: groups_) {
        if (hasAttr(group)) {
            erase(group);
        }
    }

    groups_.clear();
}

size_t mv::CompilationDescriptor::getNumGroups()
{
    return groups_.size();
}

bool mv::CompilationDescriptor::rootGroupPresent()
{
    return hasAttr("root");
}

bool mv::CompilationDescriptor::isGroup(const std::string& elem)
{
    return std::find(groups_.begin(), groups_.end(), elem) != groups_.end();
}

void mv::CompilationDescriptor::setPassArg(const std::string& pass, const std::string& arg, const mv::Attribute& value)
{
    bool found = false;

    for (auto group: groups_)
    {
        if (hasAttr(group))
        {
            auto& g = get<Element>(group);
            if (g.hasAttr(pass))
            {
                auto& p = g.get<Element>(pass);
                p.set(arg, value);

                if (!found)
                    found = true;
            }
        }
    }

    if (!found)
        throw RuntimeError(*this, "Trying to set arguments for a non-existent pass (" + pass + ")");
}

void mv::CompilationDescriptor::setPassArg(const std::string& group, const std::string& pass, const std::string& arg, const mv::Attribute& value)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to set pass arguments for a non-existent group (" + group + ")");

    Element& g = get<Element>(group);

    if (!g.hasAttr(pass))
        throw RuntimeError(*this, "Trying to set arguments for a non-existent pass (" + pass + ") in group (" + group + ")");

    Element& p = g.get<Element>(pass);
    p.set(arg, value);
}

mv::Attribute mv::CompilationDescriptor::getPassArg(const std::string& group, const std::string& pass, const std::string& arg)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to get arguments from a non-existent group (" + pass + ")");

    auto& g = get<Element>(group);
    if (!g.hasAttr(pass))
        throw RuntimeError(*this, "Trying to retrieve arguments for a non-existent pass (" + pass + ") in group (" + group + ")");

    auto& p = g.get<Element>(pass);

    if (!p.hasAttr(arg))
        throw RuntimeError(*this, "Trying to retrieve arg value for a non-existent argument (" + arg + ")");

    return p.get(arg);
}

void mv::CompilationDescriptor::serializePassListInGroup(const std::string& group, std::vector<Element *>& serializedPasses)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to serialize passes in a non-existent group (" + group + ")");

    Element &elem = get<mv::Element>(group);

    std::vector<Element *> recurrentPasses;
    if (elem.hasAttr("Recurrent"))
    {
        std::vector<std::string> &recurrent_group = elem.get<std::vector<std::string>>("Recurrent");
        for (auto rec_elem: recurrent_group)
        {
            if (isGroup(rec_elem))
                serializePassListInGroup(rec_elem, recurrentPasses);
            else
            {
                auto& pass = elem.get<Element>(rec_elem);
                recurrentPasses.push_back(&pass);
            }
        }
    }

    if (elem.hasAttr("Singular"))
    {
        std::vector<std::string> &singular_group = elem.get<std::vector<std::string>>("Singular");
        for (auto sing_elem: singular_group)
        {
            if (isGroup(sing_elem))
                serializePassListInGroup(sing_elem, serializedPasses);
            else
            {
                auto& pass = elem.get<Element>(sing_elem);
                serializedPasses.push_back(&pass);
            }

            if (!recurrentPasses.empty())
                serializedPasses.insert(serializedPasses.end(), recurrentPasses.begin(), recurrentPasses.end());
        }
    }
}

std::vector<mv::Element> mv::CompilationDescriptor::serializePassList()
{

    if (!hasAttr("root"))
        throw RuntimeError(*this, "Unable to find root group, cannot serialize passes in the descriptor.");

    std::vector<Element *> serialized;

    serializePassListInGroup("root", serialized);

    std::vector<Element> serializedPasses;
    for (auto p: serialized) {
        serializedPasses.push_back(*p);
    }

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
