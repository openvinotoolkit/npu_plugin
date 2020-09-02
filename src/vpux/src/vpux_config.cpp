#include "vpux_config.hpp"

#include <vpu/vpu_compiler_config.hpp>

void vpux::VPUXConfigBase::parse(const std::map<std::string, std::string>& config) {
    ParsedConfigBase::parse(config);
    for (const auto& p : config) {
        auto it = _config.find(p.first);
        // FIXME: insert_or_assign (c++17)
        if (it != _config.end()) {
            it->second = p.second;
        } else {
            _config.insert(p);
        }
    }
}

vpux::VPUXConfigBase::VPUXConfigBase(): _options(vpu::ParsedConfigBase::getCompileOptions()) {}

void vpux::VPUXConfigBase::expandSupportedOptions(const std::unordered_set<std::string>& options) {
    _options.insert(options.begin(), options.end());
}

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getCompileOptions() const { return _options; }

vpux::VPUXConfig::VPUXConfig() {
    _options = merge(vpux::VPUXConfigBase::getCompileOptions(), {
                                                                    VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER),
                                                                });
}

void vpux::VPUXConfig::parseFrom(const vpux::VPUXConfig& other) { parse(other.getConfig()); }

void vpux::VPUXConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfigBase::parse(config);
    setOption(_useNGraphParser, switches, config, VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER));
}
