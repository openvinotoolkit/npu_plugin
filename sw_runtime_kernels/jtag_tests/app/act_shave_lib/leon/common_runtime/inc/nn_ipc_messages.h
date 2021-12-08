/*
 * {% copyright %}
 */
#ifndef NN_IPC_MESSAGES_H_
#define NN_IPC_MESSAGES_H_

namespace nn {
namespace ipc {
namespace messages {
// LRT -> LNN
struct alignas(4) ShutdownRequest {};

struct alignas(8) ResourceLimitRequest {
    unsigned int dpu_mask_;
};

// LNN -> LRT
struct alignas(16) StartupNotification {
    void *staticMapping_;
#ifdef NN_ENABLE_SCALABILITY_REPORTING
    void *nnLogBuffer_;
#endif /* NN_ENABLE_SCALABILITY_REPORTING */
};

struct alignas(8) ResourceLimitNotification : public ResourceLimitRequest {};

struct alignas(4) ShutdownNotification {};
} // namespace messages
} // namespace ipc
} // namespace nn

#endif // NN_IPC_MESSAGES_H_
