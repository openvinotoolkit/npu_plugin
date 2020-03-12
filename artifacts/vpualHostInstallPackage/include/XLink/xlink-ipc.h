/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * XLink Linux Kernel API
 *
 * Copyright (C) 2018-2019 Intel Corporation
 *
 */
#ifndef __XLINK_IPC_API_H
#define __XLINK_IPC_API_H
#if 1
enum XLinkProtocol {
	USB_VSC = 0,
	USB_CDC,
	PCIE,
	IPC,
	SPI,
	ETH,
	NMB_OF_PROTOCOLS
};

enum XLinkDeviceTypes {
	VPU_DEVICE,
	PCIE_DEVICE,
	USB_DEVICE,
	ETH_DEVICE,
	IPC_DEVICE,
	NMB_OF_DEVICE_TYPES
};

struct XLinkHandler {
	char *devicePath;
	char *devicePath2;
	int linkId;
	uint8_t  node;
	enum XLinkDeviceTypes deviceType;
};

enum OperationMode {
	RXB_TXB,
	RXN_TXN,
	RXB_TXN,
	RXN_TXB
};

enum XLinkError {
	X_LINK_SUCCESS = 0,
	X_LINK_CHAN_ALREADY_OPEN,
	X_LINK_CHAN_NOT_OPEN,
	X_LINK_COMMUNICATION_NOT_OPEN,
	X_LINK_COMMUNICATION_FAIL,
	X_LINK_COMMUNICATION_UNKNOWN_ERROR,
	X_LINK_DEVICE_NOT_FOUND,
	X_LINK_TIMEOUT,
	X_LINK_INVALID_CHAN,
	X_LINK_INVALID_PARAM,
	X_LINK_DATA_EXCEEDS_MAX,
	X_LINK_OUT_OF_MEMORY,
	X_LINK_VPU_STOP_FAIL,
	X_LINK_VPU_NOT_STOPPED,
	X_LINK_VPU_START_FAIL,
	X_LINK_VPU_NO_READY_MSG,
	X_LINK_VPU_NOT_READY,
	X_LINK_ERROR
};

enum ChannelStatus {
	CHAN_CLOSED			= 0x00,
	CHAN_OPEN			= 0x01,
	CHAN_BLOCKED_READ	= 0x10,
	CHAN_BLOCKED_WRITE	= 0x100,
};
#endif
enum XLinkError _xlink_open_channel(struct XLinkHandler *deviceHandler,
		uint16_t chan, enum OperationMode opMode, uint32_t dataSize,
		uint32_t timeout);

enum XLinkError _xlink_close_channel(struct XLinkHandler *deviceHandler,
		uint16_t chan);
enum XLinkError _XLinkWriteData(struct XLinkHandler *deviceHandler,
		uint16_t chan,
		uint32_t  *message, uint32_t size);

enum XLinkError _XLinkReadData(struct XLinkHandler *deviceHandler,
		uint16_t chan,
		uint32_t *message, uint32_t *size);

enum XLinkError _XLinkWriteCrcData(struct XLinkHandler *deviceHandler,
		uint16_t chan, const uint8_t *message, uint32_t size);

enum XLinkError _xlink_write_volatile(struct XLinkHandler *deviceHandler,
		uint16_t chan, uint32_t *volatileMessage, uint32_t size);

enum XLinkError _XLinkReadDataToBuffer(struct XLinkHandler *deviceHandler,
		uint16_t chan, uint8_t * const message, uint32_t *size);

enum XLinkError _XLinkReadCrcDataToBuffer(struct XLinkHandler *deviceIdHandler,
		uint16_t chan, uint8_t * const message, uint32_t *size);

enum XLinkError _XLinkReleaseData(uint16_t chan);

enum XLinkError _XLinkStartVpu(char *filename);
enum XLinkError _XLinkStopVpu(void);

#endif /* __XLINK_IPC_API_H */
