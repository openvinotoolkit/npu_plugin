// {% copyright %}

#pragma once

#include <mv_types.h>
#include <nn_context.h>
#include <nn_resource_locator.h>
#include <nn_resource_manager.h>
#include <Dma.h>

namespace nn {
namespace inference_runtime {
namespace context {

using namespace util;
using namespace common_runtime;
using namespace inference_context;

/// partitions shave L2 for act-shaves to share the same cache slice
void configure_nce_shave_l2_for_single_user_context();

/// partitions shave L2 for act-shaves of different context to use separate cache slices
void configure_nce_shave_l2_for_user_context_per_tile();

/// sets up NCE for single user context and initializes internal memory state
void init_context_default_state(StaticMapping &sm);

/// @returns true if tiles and L2 are set up to support a single user context
bool configured_for_single_context();

/// gets mapping of 5bit context ID to 20bit HostSubstreamID used by MMU
uint32_t get_host_ID_mapping(uint32_t context_id);

/// gets the tile mask associated with the context
ResourceMask get_bound_resources(uint32_t context_id);

/// gets the id associated with slice
uint8_t get_context_id(uint8_t tile);

/// this leaves act-shaves for tile disabled
bool flush_tile_of_context(StaticMapping &sm, uint8_t tile);

/// a convenience function to operate on all tiles. This leaves act-shaves for tiles disabled
void flush_tiles_of_context(StaticMapping &sm);

/// reconfigures tile for context, enables act-shaves
bool prepare_tile_for_context(uint8_t tile, uint32_t context_id, StaticMapping &sm);

/// a convenience function to operate on all tiles reconfigures tiles for context, enables act-shaves
bool prepare_tiles_for_context(uint32_t context_id, StaticMapping &sm);

/// records context violation sticky registers to local memory
bool check_and_record_context_violation(ViolationState &);

/// prints any detected NCE firewall violations as WARNING
void print_context_violation(ViolationState &);

/// Enable CID IRQs. Bit per context ID
void context_violation_irq_enable(uint32_t cid_mask);

/// Clear pending context violation for specified CID. Bit per CID
void context_violation_irq_clear(uint32_t cid_mask);

/// Wipe a CMX tile allocated to a user-context
DmaStatus wipe_cmx(char tile, uint32_t dest, uint32_t size);

} // namespace context
} // namespace inference_runtime
} // namespace nn
