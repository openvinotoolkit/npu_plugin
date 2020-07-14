Built from https://github.com/movidius/vpuip_2 263d40ed3db679720a35c9b0662f9bb544db9da7 tag: NN_Runtime_v2.41.3

```
cd vpuip_2/application/vpuFirmware
python3.7 make_std_fw_image.py -a FW_bootLoader -o vpu.bin -fva 9eed6e16220cdc5c5c74d29ee9ee40593f340a25 -fla 0x84802000 -fcla 0x84800000 -fvla 0x84801000 --run-target kmb_silicon
```

It is possible to choose the firmware via environment variable VPU_FIRMWARE_FILE:
* if this variable is not set, /lib/firmware/vpu.bin is used
* if this variable is set to "vpu_custom.bin", /lib/firmware/vpu_custom.bin is used
* if this variable is set to empty string (VPU_FIRMWARE_FILE=""), firmware must be loaded with moviDebug
