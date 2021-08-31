/*
 * {% copyright %}
 */
#if 0
#define NN_ACT_ARGS_H_

/**
 * @brief These arguments are provided by the compiler and are produced in its entirety per activation kernel
 * invocation. It is expected that the schedule copies args to a location (CMX or DDR) shared with the kernel's .data
 * section in order to share a single window indirection. These args are passed by reference to the kernel by the Act
 * Runtime.
 */
typedef struct {
    char *input;
    char *output;

    unsigned int tensor_size;
    unsigned int tensor_size_x;
    unsigned int tensor_size_y;
    unsigned int tensor_size_z;

    // TODO: we'll need something like: `uint32_t[] tensor_dims;`

    /// These can all be zero, but then all invocations of a kernel must have a unique input/output pointers
    unsigned int global_input_offset;
    unsigned int local_input_offset;
    unsigned int global_output_offset;
    unsigned int local_output_offset;

    /**
     * A pointer to the kernel task's kernel-specific args. This datatype is defined with the kernel implementation. It
     * is global to all invocations within a kernel range.
     */
    const void *global_custom_args;
    unsigned int gca_size;
} act_kernel_args;

#endif
