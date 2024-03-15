//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux_private_properties.hpp"

namespace vpux {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

class IMDInferRequest final : public SyncInferRequest {
public:
    using Ptr = std::shared_ptr<IMDInferRequest>;

    explicit IMDInferRequest(const std::shared_ptr<const ov::ICompiledModel> compiledModel,
                             const std::shared_ptr<const NetworkDescription> networkDescription,
                             const Executor::Ptr executor, const Config& config);

    void infer() override;
    void infer_async() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    void get_result() override;

private:
    void check_network_precision(const ov::element::Type_t precision) override;

    SmallString create_temporary_work_directory();
    void store_compiled_model();
    void store_network_inputs();
    void run_app();
    void read_from_file(const std::string& path, const std::shared_ptr<ov::ITensor>& tensor);
    void load_network_outputs();

    SmallString _workDirectory;
    const Executor::Ptr _executorPtr;
    const Config _config;
    Logger _logger;

    std::unordered_map<std::string, size_t> _inputOrder;
    std::unordered_map<std::string, size_t> _outputOrder;

    std::shared_ptr<ov::ITensor> _rawProfilingData;
};

namespace profiling {

LayerStatistics getLayerStatistics(const uint8_t* rawData, size_t dataSize, const std::vector<char>& blob);

}  //  namespace profiling

}  //  namespace vpux
