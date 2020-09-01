#include "vpual_device.hpp"

#include <memory>

#include "vpusmm_allocator.hpp"

namespace vpux {

namespace {
// expected format VPU-#, where # is device id
int extractIdFromDeviceName(const std::string& name) {
    const std::size_t expectedSize = 5;
    if (name.size() != expectedSize) {
        THROW_IE_EXCEPTION << "Unexpected device name: " << name;
    }

    return name[expectedSize - 1] - '0';
}
}  // namespace

VpualDevice::VpualDevice(const std::string& name): _name(name) {
    const auto id = extractIdFromDeviceName(name);
    _allocator = std::make_shared<VpusmmAllocator>(id);
}

std::shared_ptr<Allocator> VpualDevice::getAllocator() const { return _allocator; }

std::string VpualDevice::getName() const { return _name; }
}  // namespace vpux
