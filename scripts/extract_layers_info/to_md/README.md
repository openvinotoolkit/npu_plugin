`to_md.py` scripts are intended to present layer information 
from several IRs xml files as tree of markdown formatted files 

Required Python 3.5 or higher

Script originally was based on script info.py by @atarakan . (https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin/merge_requests/360)

Usage:
```
python3 to_md.py <path to root directory with IRs> <path to directory for results>
```
`to_md.py` results should be uploaded to gitlab repository to be viewed

[Example](https://gitlab-icv.inn.intel.com/inference-engine/models-ir/blob/9e07ab6dbb671d1c97630ac7862e298329ed717b/KMB_models/INT8/LAYERS_NETWORKS.mdd) 
to look at and walk over one of the results of the script to_md.py

