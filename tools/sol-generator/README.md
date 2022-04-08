# Speed of light IR generator

## Summary

Generator for Speed of Light (SoL) models, which is focused on testing E2E latency without taking into account computation (have minimum amount of computation logic and data transition on device).

Example:

```cmd
sol-generator.exe -output SoL_isize-2048-osize-1024 -inputs_size "2048" -outputs_size "1024"
sol-generator.exe -output SoL_MIMO-1024 -inputs_size "1024 2048" -outputs_size "1024 2048"
```
