// RUN: vpux-opt %s --init-compiler="vpu-arch=KMB" --init-compiler="vpu-arch=KMB" -verify-diagnostics

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @test {
}
