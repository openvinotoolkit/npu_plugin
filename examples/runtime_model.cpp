#include "include/mcm/compiler/runtime/runtime_model.hpp"

int main()
{
    mv::RuntimeModel model;
    flatbuffers::FlatBufferBuilder builder;
    auto serializedModel = convertToFlatbuffer(&model, builder);
    builder.Finish(serializedModel);
    uint8_t *buf = builder.GetBufferPointer();
    int size = builder.GetSize(); // Returns the size of the buffer that
                                  // `GetBufferPointer()` points to.
    return 0;
}
