/*
 * {% copyright %}
 */
#ifndef NN_RING_BUFFER_H_
#define NN_RING_BUFFER_H_

#include <algorithm>

namespace nn {
namespace util {
template <typename S, typename T = typename S::value_type>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t size)
        : storage_(size)
        , put_(size)
        , get_(0)
        , count_(0) {}

    RingBuffer()
        : storage_()
        , put_(storage_.size())
        , get_(0)
        , count_(0) {}

    ~RingBuffer() {}

    void push(const T &t) {
        ++count_;
        if (++put_ >= capacity())
            put_ = 0;
        storage_[put_] = t;
    }

    void pop() {
        --count_;
        if (++get_ >= capacity())
            get_ = 0;
    }

    void clear() {
        put_ = capacity();
        get_ = 0;
        count_ = 0;
    }

    T &front() { return storage_[get_]; }

    const T &front() const { return storage_[get_]; }

    T &back() { return storage_[put_]; }

    const T &back() const { return storage_[put_]; }

    inline bool empty() const { return size() == 0; }

    inline bool full() const { return size() == capacity(); }

    inline std::size_t size() const { return count_; }

    inline std::size_t capacity() const { return storage_.size(); }

private:
    S storage_;
    unsigned int put_;
    unsigned int get_;
    unsigned int count_;

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
};
} // namespace util
} // namespace nn

#endif // NN_RING_BUFFER_H_
