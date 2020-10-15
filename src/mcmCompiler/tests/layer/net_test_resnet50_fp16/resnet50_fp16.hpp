#pragma once

#include <string>

#include "mcm/compiler/compilation_unit.hpp"

mv::CompilationUnit buildResnet50_fp16(const std::string& binaryDir);
