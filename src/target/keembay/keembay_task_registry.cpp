// Define base tasks/OPs //
#include    "src/computation/op/op_registry.cpp"

// Keembay specific tasks //

#include    "src/target/keembay/tasks/def/barrier_task.cpp"
#include    "src/target/keembay/tasks/def/deallocate_task.cpp"
#include    "src/target/keembay/tasks/def/dma_task.cpp"
#include    "src/target/keembay/tasks/def/dpu_task.cpp"
#include    "src/target/keembay/tasks/def/placeholder_task.cpp"
#include    "src/target/keembay/tasks/def/sparsity_map.cpp"
#include    "src/target/keembay/tasks/def/weights_table.cpp"
#include    "src/target/keembay/tasks/def/implicit_concat.cpp"
