/*
Copyright (C) 2019 Intel Corporation

SPDX-License-Identifier: MIT
*/

#ifndef _RESMGR_H_
#define _RESMGR_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CHANNEL_TYPE {
    RESMGR_PCIE_CHANNEL   = 1,
    RESMGR_MAX_TYPE
};
/** allocate one xlink channel based on the channel type
 * @param channnel_name [IN]: The channel name should be unique string specified by invoker.
                              The channel name should point to the NULL-end buffer. The string
                              length can't exceed 40 bytes;
 * @param channel_type [IN]: The type of channel which invoker want to allocate.
 * @param sw_device_id [IN]: The sw_device_id which is loacated in the structure of xlink core handle.
			     The user need to ensure the sw_device_id parameter is correct;
 * @param channel_id [OUT]: The pointer to the channel_id;
 * @return: 0 for success of allocating;
 *          EEXIST if the channel has been allocated already;
 *          Other general errno in linux;
 * The channel is recorded as allocated after this API returns with success. The user needs to maintain
 * channel open/close() in application level.
 */
int32_t cm_allocate_channel(const char* channel_name, enum CHANNEL_TYPE channel_type,  int32_t sw_device_id, uint16_t *channel_id);
/** deallocate one xlink channel based on the channel id;
 * @param sw_device_id [IN]: The sw_device_id which is loacated in the structure of xlink core handle.
			     The user need to ensure the sw_device_id parameter is correct;
 * @param channel_id [IN]: The channel id which invoker want to deallocate;
 * @return: 0 for success others for failure.
 * The channel is deallocated in allocator after this API returns with success. The user needs to maintain
 * channel open/close() in application level.
 */
int32_t cm_deallocate_channel (int32_t sw_device_id, uint16_t channel_id);

#ifdef __cplusplus
}
#endif

#endif
