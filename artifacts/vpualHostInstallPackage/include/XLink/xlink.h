///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
// SPDX-License-Identifier: GPL-2.0-only
/*
 * xlink Linux Kernel API
 *
 * Copyright (C) 2018-2019 Intel Corporation
 *
 */
#ifndef __XLINK_H
#define __XLINK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t xlink_channel_id_t;

enum xlink_prof_cfg {
	PROFILE_DISABLE = 0,
	PROFILE_ENABLE
};

enum xlink_dev_type {
	VPU_DEVICE = 0,
	PCIE_DEVICE,
	USB_DEVICE,
	ETH_DEVICE,
	IPC_DEVICE,
	NMB_OF_DEVICE_TYPES
};

struct xlink_handle {
	uint32_t link_id;
	uint32_t sw_device_id;
	enum xlink_dev_type dev_type;
	void *fd;
	uint8_t node;
};

enum xlink_opmode {
	RXB_TXB = 0,
	RXN_TXN,
	RXB_TXN,
	RXN_TXB
};

enum xlink_sys_freq {
	DEFAULT_NOMINAL_MAX = 0,
	POWER_SAVING_MEDIUM,
	POWER_SAVING_HIGH,
};

enum xlink_error {
	X_LINK_SUCCESS = 0,
	X_LINK_ALREADY_INIT,
	X_LINK_ALREADY_OPEN,
	X_LINK_COMMUNICATION_NOT_OPEN,
	X_LINK_COMMUNICATION_FAIL,
	X_LINK_COMMUNICATION_UNKNOWN_ERROR,
	X_LINK_DEVICE_NOT_FOUND,
	X_LINK_TIMEOUT,
	X_LINK_ERROR,
	X_LINK_CHAN_FULL
};

enum xlink_channel_status {
	CHAN_CLOSED			= 0x0000,
	CHAN_OPEN			= 0x0001,
	CHAN_BLOCKED_READ	= 0x0010,
	CHAN_BLOCKED_WRITE	= 0x0100,
	CHAN_OPEN_PEER		= 0x1000,
};

enum xlink_device_status {
	XLINK_DEV_OFF,
	XLINK_DEV_ERROR,
	XLINK_DEV_BUSY,
	XLINK_DEV_RECOVERY,
	XLINK_DEV_READY
};

enum xlink_error xlink_initialize(void);

enum xlink_error xlink_connect(struct xlink_handle *handle);

enum xlink_error xlink_open_channel(struct xlink_handle *handle,
		uint16_t chan, enum xlink_opmode mode, uint32_t data_size,
		uint32_t timeout);

enum xlink_error xlink_data_ready_callback(struct xlink_handle *handle,
		uint16_t chan, void *func);

enum xlink_error xlink_data_consumed_callback(struct xlink_handle *handle,
		uint16_t chan, void *func);

enum xlink_error xlink_close_channel(struct xlink_handle *handle,
		uint16_t chan);

enum xlink_error xlink_write_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t const *message, uint32_t size);

enum xlink_error xlink_read_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t **message, uint32_t *size);

enum xlink_error xlink_write_control_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t const *message, uint32_t size);

enum xlink_error xlink_write_volatile(struct xlink_handle *handle,
		uint16_t chan, uint8_t const *message, uint32_t size);

enum xlink_error xlink_read_data_to_buffer(struct xlink_handle *handle,
		uint16_t chan, uint8_t * const message, uint32_t *size);

enum xlink_error xlink_release_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t * const data_addr);

enum xlink_error xlink_disconnect(struct xlink_handle *handle);

enum xlink_error xlink_get_device_name(uint32_t sw_device_id, char *name,
		size_t name_size);

enum xlink_error xlink_get_device_list(uint32_t *sw_device_id_list,
		uint32_t *num_devices, int pid);

enum xlink_error xlink_get_device_status(uint32_t sw_device_id,
		uint32_t *device_status);

enum xlink_error xlink_boot_device(struct xlink_handle *handle,
		enum xlink_sys_freq operating_frequency);

enum xlink_error xlink_reset_device(struct xlink_handle *handle,
		enum xlink_sys_freq operating_frequency);

enum xlink_error xlink_start_vpu(char *filename);

enum xlink_error xlink_stop_vpu(void);

/* API functions to be implemented

enum xlink_error xlink_write_crc_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t const *message, size_t size);

enum xlink_error xlink_read_crc_data(struct xlink_handle *handle,
		uint16_t chan, uint8_t **message, size_t * const size);

enum xlink_error xlink_read_crc_data_to_buffer(struct xlink_handle *handle,
		uint16_t chan, uint8_t * const message, size_t * const size);

enum xlink_error xlink_reset_all(enum xlink_sys_freq operating_frequency);

 */

#ifdef __cplusplus
}
#endif

#endif /* __XLINK_H */
