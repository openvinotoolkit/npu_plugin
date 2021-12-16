/*
 * {% copyright %}
 */
#ifndef NN_SHARED_BUFFER_H_
#define NN_SHARED_BUFFER_H_

#include <mv_types.h>
#include <algorithm>
#include <string.h>

namespace nn {
namespace util {
class SharedBuffer {
public:
    enum Process { CONSUMER, PRODUCER };

    SharedBuffer(uint32_t *buffer, uint32_t length, uint32_t elemSize, Process process)
        : ctrl_(buffer)
        , nextHead_(0)
        , nextTail_(0)
        , elemSize_(elemSize) {
        // Producer initializes the control data after memory has been allocated
        if (process == PRODUCER)
            initCtrlData(length);
    };

    SharedBuffer(){};
    ~SharedBuffer(){};
    SharedBuffer(const SharedBuffer &) = delete;
    SharedBuffer &operator=(const SharedBuffer &) = delete;

    bool updateProducer(void) {
        // Check if space available
        if (!space())
            return false;

        // Find available space to write, wrapping if needed
        if (reinterpret_cast<uint32_t>(*ctrl_.tail_) + elemSize_ > reinterpret_cast<uint32_t>(*ctrl_.end_))
            nextTail_ = *ctrl_.start_;
        else
            nextTail_ = *ctrl_.tail_;

        return true;
    };

    bool updateConsumer(void) {
        // Check if enough bytes are in the buffer
        if (!(entries()))
            return false;

        // Find available space to read, wrapping if needed
        if (reinterpret_cast<uint32_t>(*ctrl_.head_) + elemSize_ > reinterpret_cast<uint32_t>(*ctrl_.end_))
            nextHead_ = *ctrl_.start_;
        else
            nextHead_ = *ctrl_.head_;

        return true;
    };

    void updateTail() { *ctrl_.tail_ = nextTail_ + elemSize_; };

    void updateHead() { *ctrl_.head_ = nextHead_ + elemSize_; };

    uint32_t ctrlData() { return ctrlDataSize_; };
    uint32_t *ctrl() { return ctrl_.head_; };
    uint32_t *head() { return reinterpret_cast<uint32_t *>(nextHead_); };
    uint32_t *tail() { return reinterpret_cast<uint32_t *>(nextTail_); };
    uint32_t length() { return reinterpret_cast<uint32_t>(ctrl_.length_); };

private:
    struct Ctrl {
        uint32_t *const head_;
        uint32_t *const tail_;
        uint32_t *const start_;
        uint32_t *const end_;
        uint32_t *const length_;
        void *const data_;

        Ctrl()
            : head_(0)
            , tail_(0)
            , start_(0)
            , end_(0)
            , length_(0)
            , data_(0) {}

        Ctrl(uint32_t *buffer)
            : head_(buffer + offsetof(Ctrl, head_) / sizeof(uint32_t *))
            , tail_(buffer + offsetof(Ctrl, tail_) / sizeof(uint32_t *))
            , start_(buffer + offsetof(Ctrl, start_) / sizeof(uint32_t *))
            , end_(buffer + offsetof(Ctrl, end_) / sizeof(uint32_t *))
            , length_(buffer + offsetof(Ctrl, length_) / sizeof(uint32_t *))
            , data_(buffer + offsetof(Ctrl, data_) / sizeof(uint32_t *)) {}
    };

    Ctrl ctrl_;
    uint32_t nextHead_;
    uint32_t nextTail_;
    uint32_t elemSize_;
    static constexpr uint32_t ctrlDataSize_ = offsetof(Ctrl, data_);

    void initCtrlData(uint32_t length) {
        // Set the head, tail and start pointers to the start of the data section
        *ctrl_.head_ = *ctrl_.tail_ = *ctrl_.start_ = reinterpret_cast<uint32_t>(ctrl_.data_);

        // Make the usable buffer length an even multiple of the written data structure, to
        // simplify the queue wrap logic
        *ctrl_.length_ = (length - ctrlDataSize_) - ((length - ctrlDataSize_) % elemSize_);
        *ctrl_.end_ = *ctrl_.start_ + *ctrl_.length_;
    };

    int space() { return (*ctrl_.length_ / elemSize_) - entries() - 1; };

    int entries(void) {
        if (*ctrl_.tail_ >= *ctrl_.head_)
            return (*ctrl_.tail_ - *ctrl_.head_) / elemSize_;
        else
            return (*ctrl_.length_ + *ctrl_.tail_ - *ctrl_.head_) / elemSize_;
    };

    void printCtrl(void) {
        nnLog(MVLOG_DEBUG, "&ctrl_.head_   = %p, *ctrl_.head_   = %x", ctrl_.head_, *ctrl_.head_);
        nnLog(MVLOG_DEBUG, "&ctrl_.tail_   = %p, *ctrl_.tail_   = %x", ctrl_.tail_, *ctrl_.tail_);
        nnLog(MVLOG_DEBUG, "&ctrl_.start_  = %p, *ctrl_.start_  = %x", ctrl_.start_, *ctrl_.start_);
        nnLog(MVLOG_DEBUG, "&ctrl_.end_    = %p, *ctrl_.end_    = %x", ctrl_.end_, *ctrl_.end_);
        nnLog(MVLOG_DEBUG, "&ctrl_.length_ = %p, *ctrl_.length_ = %x", ctrl_.length_, *ctrl_.length_);
        nnLog(MVLOG_DEBUG, "&ctrl_.data_   = %p", ctrl_.data_);
        nnLog(MVLOG_DEBUG, "ctrlDataSize_  = %p", ctrlDataSize_);
        nnLog(MVLOG_DEBUG, "elemSize_      = %x", elemSize_);
    };
};
} // namespace util
} // namespace nn

#endif // NN_SHARED_BUFFER_H_
