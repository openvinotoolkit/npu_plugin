#include "include/mcm/base/json/object.hpp"

mv::json::Object::Object()
{
    
}

/*bool mv::json::Object::emplace(const std::string& name, const Value& value)
{
    if (members_.find(name) == members_.end())
    {
        members_.emplace(name, std::make_unique);
        return true;
    }   

    return false;

}

bool mv::json::Object::removeMember(const std::string& name)
{

    if (members_.find(name) != members_.end())
    {
        members_.erase(name);
        return true;
    }   

    return false;

}*/

/*mv::json::Value& mv::json::Object::getMember(const std::string& name)
{

    if (members_.find(name) == members_.end())
        throw std::out_of_range("Object " + name_ + " does not have member " + name);
    
    return members_[name];

}*/

/*std::map<std::string, mv::json::Value>::iterator mv::json::Object::begin()
{
    return members_.begin();
}

std::map<std::string, mv::json::Value>::iterator mv::json::Object::end()
{
    return members_.end();
}*/

unsigned mv::json::Object::size() const
{
    return members_.size();
}
