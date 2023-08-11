#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPURegMapped {

size_t calcMinBitsRequirement(uint64_t value) {
    if (value == 0) {
        return 1;
    }
    if (value == std::numeric_limits<uint64_t>::max()) {
        return sizeof(uint64_t) * CHAR_BIT;
    }
    return checked_cast<size_t>(std::ceil(log2(value + 1)));
}

void updateRegMappedInitializationValues(std::map<std::string, std::map<std::string, uint64_t>>& values,
                                         const std::map<std::string, std::map<std::string, uint64_t>>& newValues) {
    for (auto newRegisterIter = newValues.begin(); newRegisterIter != newValues.end(); ++newRegisterIter) {
        auto correspondingRegisterIter = values.find(newRegisterIter->first);
        VPUX_THROW_UNLESS(correspondingRegisterIter != values.end(),
                          "updateRegMappedInitializationValues: Register with name {0} not found in provided values",
                          newRegisterIter->first);

        for (auto newFieldIter = newRegisterIter->second.begin(); newFieldIter != newRegisterIter->second.end();
             ++newFieldIter) {
            auto correspondingFieldIter = correspondingRegisterIter->second.find(newFieldIter->first);
            VPUX_THROW_UNLESS(correspondingFieldIter != correspondingRegisterIter->second.end(),
                              "updateRegMappedInitializationValues: Field with name {0} not found in provided values",
                              newFieldIter->first);

            // update field value
            correspondingFieldIter->second = newFieldIter->second;
        }
    }
}

}  // namespace VPURegMapped
}  // namespace vpux
