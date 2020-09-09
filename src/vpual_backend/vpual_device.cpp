#include "vpual_device.hpp"

#include <memory>

#include "vpusmm_allocator.hpp"

namespace vpux {

namespace {
// expected format VPU-#, where # is device id
// can also be VPU-32 for CSRAM. this is an exceptional situation
int extractIdFromDeviceName(const std::string& name) {
    const std::size_t expectedSize = 5;
    if (name.size() != expectedSize) {
        if (name == vpux::CSRAM_SLICE_ID) {
            int deviceId = 0;
            int multiplier = 1;
            // digitPos < 2 because number 32 has only 2 digits
            for (size_t digitPos = 0; digitPos < 2; digitPos++) {
                int digit = (name[name.size() - digitPos - 1] - '0');
                deviceId += digit * multiplier;
                multiplier *= 10;
            }
            return deviceId;
        } else {
            THROW_IE_EXCEPTION << "Unexpected device name: " << name;
        }
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
