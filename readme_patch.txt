To verify the patch run:

git apply --stat adapt_for_production_runtime.patch
git apply --check adapt_for_production_runtime.patch

Once you have verified that everything is okay, you can apply with:

git am --signoff < adapt_for_production_runtime.patch

