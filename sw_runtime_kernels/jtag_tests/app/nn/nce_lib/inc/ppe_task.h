/*
* {% copyright %}
*/
#ifndef PPE_TASK_H
#define PPE_TASK_H

namespace nn
{
    namespace nce_lib
    {
        enum class FixOpcodes : unsigned char
        {
            NOOP = 0x00,
            CLEAR = 0x01,
            LOAD = 0x02,
            STORE = 0x03,
            HALT = 0x06,
            ADD = 0x08,
            SUB = 0x09,
            MULT = 0x0A,
            MAX = 0x0B,
            MIN = 0x0C,
            AND = 0x0D,
            OR = 0x0E,
            XOR = 0x0F,
            NOT = 0x10,
            ABS = 0x11,
            NEG = 0x12,
            CEIL = 0x14,
            FLOOR = 0x15,
            POW = 0x20,
            TANH = 0x22,
            SIGMOID = 0x23,
            EXP = 0x24,
            SQRT = 0x25,
            RSQRT = 0x26,
            FLEXARB = 0x27,
            BYPASS = 0x28,
            RELU = 0x29,
            LPRELU = 0x2A,
            RELUX = 0x2B,
            INVALID_OPCODE
        };

        enum class RsDtype : unsigned char
        {
            FP16,
            U8F,
            G8,
            I8,
            I32,
            S1616,
            INVALID_DTYPE
        };

        enum class RdDtype : unsigned char
        {
            FP16,
            U8F,
            G8,
            I8,
            I32,
            I4,
            I2,
            LOG,
            BIN,
            INVALID_DTYPE
        };
    }
}

#endif /* PPE_TASK_H */
