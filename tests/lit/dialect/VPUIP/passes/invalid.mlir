// RUN: vpux-opt %s --set-compile-params="vpu-arch=KMB" --set-compile-params="vpu-arch=KMB" -verify-diagnostics

// -----
// expected-error@+1 {{Architecture is already defined. Probably you don't need to run '--set-compile-params'.}}
module @test {
}