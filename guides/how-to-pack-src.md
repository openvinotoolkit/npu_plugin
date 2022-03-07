# How pack source package

Pack source package using CPack:

1. Run CMake configuration stage, for example refer to the instructions in `guides/how-to-build-static.md`
2. Run CPack to pack the source package
```bash
cpack -G ZIP --config %VPUX_PLUGIN_HOME%/build-x86_64/CPackSourceConfigVPUX.cmake
```
