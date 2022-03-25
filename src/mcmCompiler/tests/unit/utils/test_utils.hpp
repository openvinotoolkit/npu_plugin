//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#pragma once

#include "mcm/tensor/shape.hpp"
#include "mcm/target/target_descriptor.hpp"

#include <string>

std::string testToString(const mv::Shape &shape);
std::string testToString(const mv::Target& target);
