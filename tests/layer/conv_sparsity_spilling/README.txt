Test Plan:
--------
This is a simple test where we force the activation (output) to be sparse and
forcefully spill it to DDR and read it back to a different address in the CMX.

TEST:
-----
To exercise the test plan use the CD `force_sparse_activation_spill.json` as
follows:

$ ./conv_sparsity_spilling force_sparse_activation_spill.json

Running the blob on EVM should not change the results.
