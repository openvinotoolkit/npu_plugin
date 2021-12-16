/*
 * {% copyright %}
 */
#ifndef NN_CACHED_SHARED_BUFFER_H_
#define NN_CACHED_SHARED_BUFFER_H_

#include <mv_types.h>
#include <nn_runtime_types.h>
#include <nn_cache.h>
#include <array>
#include <algorithm>
#include <nn_shared_buffer.h>
#include <nn_inference_runtime_types.h>

namespace nn {
namespace common_runtime {
template <typename T>
class CachedSharedBuffer {
public:
    CachedSharedBuffer(uint32_t *buffer, uint32_t size, nn::util::SharedBuffer::Process process)
        : buffer_(buffer, size, sizeof(T), process) {
        // Producer is responsible for managing the head & tail pointers
        // which it must flush to memory after initialization
        //
        // Consumer must invalidate it's view
        if (process == nn::util::SharedBuffer::Process::PRODUCER)
            nn::cache::flush(buffer_.ctrl(), size);
        else
            nn::cache::invalidate(buffer_.ctrl(), size);
    };

    CachedSharedBuffer(){};
    ~CachedSharedBuffer(){};
    CachedSharedBuffer(const CachedSharedBuffer &) = delete;
    CachedSharedBuffer &operator=(const CachedSharedBuffer &) = delete;

    uint32_t *start(void) { return buffer_.ctrl(); };
    uint32_t length(void) { return buffer_.length(); };

    bool push(T update) {
        bool rc = false;
        nn::cache::invalidate(buffer_.ctrl(), buffer_.ctrlData());

        if (buffer_.updateProducer()) {
            // Write the data & flush cache
            memcpy_s(buffer_.tail(), sizeof(T), &update, sizeof(T));
            nn::cache::flush(buffer_.tail(), sizeof(T));

            // Then update the tail & flush the control structure
            buffer_.updateTail();
            nn::cache::flush(buffer_.ctrl(), buffer_.ctrlData());

            rc = true;
        }

        return rc;
    }

    bool pop(T *update) {
        bool rc = false;
        nn::cache::invalidate(buffer_.ctrl(), buffer_.ctrlData());

        if (buffer_.updateConsumer()) {
            // Invalidate our view of the new bytes written
            nn::cache::invalidate(buffer_.head(), sizeof(T));

            // Copy here to our local buffer...
            memcpy_s(update, sizeof(T), buffer_.head(), sizeof(T));

            // ... then update head & flush the control struct
            buffer_.updateHead();
            nn::cache::flush(buffer_.ctrl(), buffer_.ctrlData());

            rc = true;
        }

        return rc;
    }

private:
    util::SharedBuffer buffer_;
};
} // namespace common_runtime
} // namespace nn

#endif /* NN_CACHED_SHARED_BUFFER_H_ */
