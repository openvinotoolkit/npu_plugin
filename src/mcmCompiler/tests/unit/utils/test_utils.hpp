#pragma once

#include "mcm/tensor/shape.hpp"
#include "mcm/target/target_descriptor.hpp"

#include <string>

std::string testToString(const mv::Shape &shape);
std::string testToString(const mv::Target& target);
