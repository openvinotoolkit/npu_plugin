Test Plan:
----------
This is a two layer network and we force the dw_conv layer to be sparse output
and force spill it to DDR and read from DDR by adding a DMA tasks in the
mv::OpModel. The serialization infrastructure will automatically add DMA tasks
for SparsityMap (SM) and StorageElement (SE) and generate a blob.

Compile the network with the following CD to force the spill:

 
$python3 -u Fathom.py generate --network-description two_layer_test_dw_conv_conv_299.tflite 
    --comp-descriptor release_kmb_force_spilled_dw_conv.json  --image sitting_cat.jpg
    --cpp --emulator

Run the blob and check for CRC match.

