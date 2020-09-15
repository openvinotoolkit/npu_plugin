Built from https://github.com/movidius/vpuip_2. Please see revisions.txt for the actual revision.

```
cd vpuip_2/application/vpuFirmware
python3.7 make_std_fw_image.py -a FW_bootLoader -o vpu_ipc.bin -fva <revision> -fla 0x84802000 -fcla 0x84800000 -fvla 0x84801000 -rt tbh_evm
```

It is possible to choose the firmware via environment variable VPU_FIRMWARE_FILE:
* if this variable is not set, /lib/firmware/vpu.bin is used
* if this variable is set to "vpu_custom.bin", /lib/firmware/vpu_custom.bin is used
* if this variable is set to empty string (VPU_FIRMWARE_FILE=""), firmware must be loaded with moviDebug
* please note, that this variable is ignored on Thunder Bay
