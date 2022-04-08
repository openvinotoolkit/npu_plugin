// RUN: vpux-opt %s --init-compiler="vpu-arch=VPUX30XX" --init-compiler="vpu-arch=VPUX30XX" -verify-diagnostics

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @test {
}
