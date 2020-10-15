This function tests the RemoveOps passs, which removes redundant operations.

Dropout and a Slice that really captures the entire input tensor are the tested operations; 
RemoveOps should remove them from the graph.

This is easiest to verify by printing dot files before and after.
