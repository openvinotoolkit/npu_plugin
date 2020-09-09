#pragma once

#include <ie_allocator.hpp>
#include <memory>
#include <string>
#include <vpux.hpp>

namespace vpux {

const std::string CSRAM_SLICE_ID = "VPU-32";

class VpualDevice final : public IDevice {
public:
    VpualDevice(const std::string& name);
    std::shared_ptr<Allocator> getAllocator() const override;

    std::string getName() const override;

private:
    std::shared_ptr<Allocator> _allocator;
    const std::string _name;
};

}  // namespace vpux
