#include "vpux_config.hpp"

#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/vpu_compiler_config.hpp>

#include "vpux_private_config.hpp"

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
    _options = merge(vpux::VPUXConfigBase::getCompileOptions(),
        {VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER), VPU_KMB_CONFIG_KEY(PLATFORM)});
}

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options =
        // FIXME: Runtime?
        merge(vpux::VPUXConfigBase::getCompileOptions(), {
                                                             KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
                                                             VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES),
                                                             VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI),
                                                             VPU_KMB_CONFIG_KEY(SIPP_OUT_COLOR_FORMAT),
                                                             VPU_KMB_CONFIG_KEY(USE_SIPP),
                                                             CONFIG_KEY(PERF_COUNT),
                                                             VPU_KMB_CONFIG_KEY(USE_M2I),
                                                             CONFIG_KEY(DEVICE_ID),
                                                         });

    return options;
}

void vpux::VPUXConfig::parseFrom(const vpux::VPUXConfig& other) { parse(other.getConfig()); }

static InferenceEngine::ColorFormat parseColorFormat(const std::string& src) {
    if (src == "RGB") {
        return InferenceEngine::ColorFormat::RGB;
    } else if (src == "BGR") {
        return InferenceEngine::ColorFormat::BGR;
    } else {
        THROW_IE_EXCEPTION << "Unsupported color format is passed.";
    }
}

void vpux::VPUXConfig::parse(const std::map<std::string, std::string>& config) {
    vpux::VPUXConfigBase::parse(config);
    setOption(_useNGraphParser, switches, config, VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER));

    setOption(_throughputStreams, config, KMB_CONFIG_KEY(THROUGHPUT_STREAMS), parseInt);

    setOption(_platform, switches, config, VPU_KMB_CONFIG_KEY(PLATFORM));

    setOption(_numberOfSIPPShaves, config, VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES), parseInt);
    IE_ASSERT(_numberOfSIPPShaves > 0 && _numberOfSIPPShaves <= 16)
        << "VpualConfig::parse attempt to set invalid number of shaves for SIPP: '" << _numberOfSIPPShaves
        << "', valid numbers are from 1 to 16";

    setOption(_SIPPLpi, config, VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI), parseInt);
    IE_ASSERT(0 < _SIPPLpi && _SIPPLpi <= 16 && vpu::isPowerOfTwo(_SIPPLpi))
        << "VpualConfig::parse attempt to set invalid lpi value for SIPP: '" << _SIPPLpi
        << "',  valid values are 1, 2, 4, 8, 16";

    setOption(_outColorFmtSIPP, config, VPU_KMB_CONFIG_KEY(SIPP_OUT_COLOR_FORMAT), parseColorFormat);
    setOption(_useSIPP, switches, config, VPU_KMB_CONFIG_KEY(USE_SIPP));
    setOption(_useM2I, switches, config, VPU_KMB_CONFIG_KEY(USE_M2I));
    setOption(_deviceId, config, CONFIG_KEY(DEVICE_ID));
}
