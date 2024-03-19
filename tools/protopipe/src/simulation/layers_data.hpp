//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>

#include "scenario/ovhelper.hpp"
#include "utils/data_providers.hpp"

std::string normalizeLayerName(const std::string& layer_name);
std::vector<cv::Mat> uploadLayerData(const std::filesystem::path& path, const std::string& tag, const LayerInfo& layer);

enum class LayersType { INPUT = 0, OUTPUT };
using LayersDataMap = std::unordered_map<std::string, std::vector<cv::Mat>>;
LayersDataMap uploadFromDirectory(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers);

LayersDataMap uploadData(const std::filesystem::path& path, const std::string& tag, const LayersInfo& layers,
                         LayersType type);

bool isDirectory(const std::filesystem::path& path);

std::vector<IDataProvider::Ptr> createConstantProviders(LayersDataMap&& layers_data,
                                                        const std::vector<std::string>& layer_names);

std::vector<IDataProvider::Ptr> createRandomProviders(const LayersInfo& layers,
                                                      const std::map<std::string, IRandomGenerator::Ptr>& generators);

std::vector<std::filesystem::path> createDirectoryLayout(const std::filesystem::path& path,
                                                         const std::vector<std::string>& layer_names);
