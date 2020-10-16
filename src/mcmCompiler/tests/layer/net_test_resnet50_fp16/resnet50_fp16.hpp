#pragma once

#include <string>

#include "mcm/compiler/compilation_unit.hpp"

void buildResnet50_fp16(mv::CompilationUnit & compilationUnit, const std::string& binaryDir);
