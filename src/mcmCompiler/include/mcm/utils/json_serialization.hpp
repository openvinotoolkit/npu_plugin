#pragma once

namespace mv {

template <typename Arg>
mv::json::Value argToJSON(const Arg& argument) {
    const auto& type = std::type_index(typeid(argument));
    const auto& toJSONFunc = mv::attr::AttributeRegistry::getToJSONFunc(type);
    return toJSONFunc(argument);
}

template <typename Arg>
Arg argFromJSON(const mv::json::Value& argument) {
    const auto& type = std::type_index(typeid(Arg));
    const auto& fromJSONFunc = mv::attr::AttributeRegistry::getFromJSONFunc(type);
    return fromJSONFunc(argument).get<Arg>();
}

}  // namespace mv
