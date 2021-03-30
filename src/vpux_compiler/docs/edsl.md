# EDSL Integration

Below are the additions required to allow EDSL code to integration into the VPUX compiler.
There is no change to the overall architecture, but simply the addition of a small number of ops and changes to the associated lowerings.

## IERT Dialect changes

A PXA op will be added to the IERT dialect that represents launching a SHAVE 'kernel' written in PXA
which will use the same typing as existing IERT ops.
The PXA op will refer to a PXA function in another module via a symbol.

```MLIR
// IERT level
%1 = IERT.PXA @kernels.eltwise_sin(%0) : (memref<100xf16>) -> (memref<100xf16>)

// PXA module
module @kernels {
  func @eltwise_sin(%in : memref<100xf16>) -> (memref<100xf16>) {
    %out = affine.parallel (%i) = (0) to (100) reduce ("assign") -> (memref<100xf16) {
      %0 = pxa.load %in[%i] : memref<100xf16>
      %1 = math.sin %0 : f16
      %2 = pxa.reduce assign %1, %in[%i] : memref<100xf16>
      affine.yield %2 : memref<100xf16>
    }
    return %out
  }
}
```

## VPUIP Dialect changes

An `EdslLayer` op will be added which takes the compiled SHAVE code for a given layer as well as the required buffers.

```MLIR
%bin_code = VPUIP.DeclarConstantTensor memref<1024xui8> = dense<...> // compiled binary code
%1 = VPUIP.EdslLayer(%bin_code, %0)
```

## EDSL to IE dialect

Eltwise ops, specials, etc will be lowered to their equivalent OP in the IE dialect.
Contractions will be lowered as **Tile dialect** ops.
The resulting `Tile` ops can exist side by side with the IE ops since they have the same typing.

## Additional IE dialect passes

If a `Tile` op is to target the DPU (i.e. convolutions), it will be recognized via a stenciling pass and converted to the appropriate `IE` Op (i.e. `ConvolutionOp`).

## IE to IERT lowering

Any remaining `Tile` ops will be lowered to the `IERT.PXA` and the appropriate PXA code will be generated and placed in the `@kernels` module.

## Additional IERT dialect passes

A 'kernel fusion' pass may be added which combines multiple `IERT.PXA` ops into a single kernel where applicable and merges the appropriate PXA functions via the existing PXA fusion pass.

## IERT to VPUIP lowering

`IERT.PXA` ops will run the existing EDSL -> SHAVE lowering pipeline to produce a binary,
and then convert the `IERT.PXA` op into a `VPUIP.EdslLayer` op.
