#include "include/mcm/compiler/compilation_descriptor.hpp"
#include "include/mcm/pass/pass_registry.hpp"

mv::json::Object mv::CompilationDescriptor::load(const std::string& filePath)
{
    JSONTextParser parser(jsonParserBufferLength_);
    json::Value jsonRoot;

    if (!parser.parseFile(filePath, jsonRoot))
    {
        throw ArgumentError("CompilationDescriptor", "filePath", filePath,
            "Unable to parse JSON file");
    }

    if (jsonRoot.valueType() != json::JSONType::Object)
        throw ArgumentError("CompilationDescriptor", "file", filePath, "malformed JSON file - cannot create compilation descriptor");

    std::vector<std::string> keys = jsonRoot.getKeys();

    for (auto key : keys)
    {
        // XXX: Assuming that all keys other than "name" will define groups
        if (key != "name")
        {
            auto convertJsonStringToObject = [key](const std::string& recType, json::Value& root)
            {
                mv::json::Value& v = root[key];

                if (v.hasKey(recType))
                {
                    mv::json::Array& a = v[recType].get<mv::json::Array>();
                    for (size_t i=0; i < a.size(); i++) {
                        if (a[i].valueType() == mv::json::JSONType::String) {
                            json::Object obj;
                            obj.emplace("name", a[i]);
                            a[i] = obj;
                        }
                    }
                }
            };

            convertJsonStringToObject("Singular", jsonRoot);
            convertJsonStringToObject("Recurrent", jsonRoot);
        }
    }

    return jsonRoot.get<json::Object>();

}

mv::CompilationDescriptor::CompilationDescriptor(const std::string& profile) :
Element("CompilationDescriptor"),
profile_(profile)
{
}

mv::CompilationDescriptor::CompilationDescriptor(const json::Object& jsonDescriptor, const std::string& profile) :
Element(jsonDescriptor, true, "CompilationDescriptor"),
profile_(profile)
{
    if (!hasAttr("root"))
        throw RuntimeError(*this, "root group missing in JSON config file, unable to construct compilation descriptor");

    // XXX: This assumes that the compilation descriptor has group definitions at the top level.
    groups_ = attrsKeys();
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
        std::vector<Element> &rec_v = g_elem.get<std::vector<Element>>(recurrence);

        if (elemIsGroup && std::find(rec_v.begin(), rec_v.end(), Element(elem)) != rec_v.end())
            throw RuntimeError(*this, "Trying to add preexisting group (" + elem + ") to group (" + group + "), which is not permitted");

        rec_v.push_back(elem);
    }
    else
    {
        Element p = Element(elem);
        p.set<bool>("isGroup", false);
        std::vector<Element> rec_v = { p };
        g_elem.set<std::vector<Element>>(recurrence, rec_v);
    }

    if (elemIsGroup && !hasAttr(elem))
        addGroup(elem);

}

void mv::CompilationDescriptor::remove(const std::string& group, const std::string& elem, const std::string& recurrence)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to remove pass from a non-existent group (" + group + ")");
    
    Element& g_elem = get<Element>(group);

    if (!g_elem.hasAttr(recurrence))
        throw RuntimeError(*this, recurrence + " recurrence is not present in group " + group);

    // Remove element from list of passes.
    auto& rec_v = g_elem.get<std::vector<Element>>(recurrence);
    
    auto it = std::find(rec_v.begin(), rec_v.end(), Element(elem));
    if (it == rec_v.end())
        throw RuntimeError(*this, "Unable to find element (" + elem + ") in group (" + group + ")");
    
    rec_v.erase(it);
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
                    auto& recList = elem.get<std::vector<Element>>(recType);
                    auto it = std::find(recList.begin(), recList.end(), Element(group));
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
    if (it != groups_.end())
        groups_.erase(it);

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
            auto addPassArgInRecType = [this, group, pass, arg, value](const std::string& recType)
            {
                auto& g = get<Element>(group);

                bool foundPass = false;
                if (g.hasAttr(recType)) {
                    std::vector<mv::Element>& recList = g.get<std::vector<Element>>(recType);

                    auto it = recList.begin();
                    while ((it = std::find (it, recList.end(), Element(pass))) != recList.end())
                    {
                        if (it->hasAttr(arg))
                            it->erase(arg);

                        it->set(arg, value);
                        foundPass = true;

                        it++;
                    }

                    if (foundPass)
                        return true;
                }
                return false;
            };

            found |= addPassArgInRecType("Singular");
            found |= addPassArgInRecType("Recurrent");
        }
    }

    if (!found)
        throw RuntimeError(*this, "Trying to set arguments for a non-existent pass (" + pass + ")");
}

void mv::CompilationDescriptor::setPassArg(const std::string& group, const std::string& pass, const std::string& arg, const mv::Attribute& value)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to set pass arguments for a non-existent group (" + group + ")");

    auto addPassArgInRecType = [this, group, pass, arg, value](const std::string& recType)
    {
        auto& g = get<Element>(group);

        if (g.hasAttr(recType)) {
            std::vector<mv::Element>& recList = g.get<std::vector<Element>>(recType);

            auto it = std::find(recList.begin(), recList.end(), Element(pass));

            if (it == recList.end())
                return false;
            else
                it->set(arg, value);

            return true;
        }

        return false;
    };

    bool f1 = addPassArgInRecType("Singular");
    bool f2 = addPassArgInRecType("Recurrent");

    if (!f1 && !f2)
        throw RuntimeError(*this, "Trying to set arguments for a non-existent pass (" + pass + ") in group (" + group + ")");

}

mv::Attribute mv::CompilationDescriptor::getPassArg(const std::string& group, const std::string& recType, const std::string& pass, const std::string& arg)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to get arguments from a non-existent group (" + pass + ")");

    auto& g = get<Element>(group);
    if (!g.hasAttr(recType))
        throw RuntimeError(*this, "Trying to retrieve arguments for a non-existent recurrence type (" + recType + ") in group (" + group + ")");

    auto& recList = g.get<std::vector<Element>>(recType);

    auto it = std::find(recList.begin(), recList.end(), Element(pass));
    if (it == recList.end())
        throw RuntimeError(*this, "Trying to retrieve arg value for a non-existent argument (" + arg + ")");
    else
    {
        if (!it->hasAttr(arg))
            throw RuntimeError(*this, "No such arg (" + arg + ") in pass (" + pass + "), in recurrence type (" + recType + ") in group (" + group + ")");
        else
            return it->get(arg);
    }
}

void mv::CompilationDescriptor::serializePassListInGroup(const std::string& group, std::vector<Element>& serializedPasses)
{
    if (!hasAttr(group))
        throw ArgumentError(*this, "CompilationDescriptor", "invalid", "Trying to serialize passes in a non-existent group (" + group + ")");

    Element &elem = get<mv::Element>(group);

    std::vector<Element> recurrentPasses;
    if (elem.hasAttr("Recurrent"))
    {
        auto& recurrent_group = elem.get<std::vector<Element>>("Recurrent");
        for (auto rec_elem: recurrent_group)
        {
            if (isGroup(rec_elem.getName()))
                serializePassListInGroup(rec_elem.getName(), recurrentPasses);
            else
                recurrentPasses.push_back(rec_elem);
        }
    }

    if (elem.hasAttr("Singular"))
    {
        auto& singular_group = elem.get<std::vector<Element>>("Singular");
        for (auto sing_elem: singular_group)
        {
            if (isGroup(sing_elem.getName()))
                serializePassListInGroup(sing_elem.getName(), serializedPasses);
            else
                serializedPasses.push_back(sing_elem);

            if (!recurrentPasses.empty())
                serializedPasses.insert(serializedPasses.end(), recurrentPasses.begin(), recurrentPasses.end());
        }
    }
}

std::vector<mv::Element> mv::CompilationDescriptor::serializePassList()
{

    if (!hasAttr("root"))
        throw RuntimeError(*this, "Unable to find root group, cannot serialize passes in the descriptor.");

    std::vector<Element> serialized;

    serializePassListInGroup("root", serialized);

    std::vector<Element> serializedPasses;
    for (auto p: serialized)
        serializedPasses.push_back(p);

    return serializedPasses;

}
