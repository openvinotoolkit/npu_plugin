#pragma once

#include "config.hpp"
#include "provider.hpp"

#include <opencv2/gapi/infer.hpp>  // GNetPackage

#include <string>
#include <vector>

class ETestsProvider : public IScenarioProvider {
public:
    ETestsProvider(const std::string& filepath, const bool use_ov_old_api);

    virtual std::vector<Scenario> createScenarios() override;

private:
    cv::gapi::GNetPackage createInferenceParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createIEParams(const std::string& tag, const Network& network);
    cv::gapi::GNetPackage createOVParams(const std::string& tag, const Network& network);

private:
    bool m_use_ov_old_api;
    ETestsConfig m_config;
};
