`to_md.py` and `to_html.py` scripts are inteded to present layer information  
from several IRs xml files as tree of markdown or html formatted files 

Scripts originally were based on script info.py by @atarakan . (https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin/merge_requests/360)

Usage:
```
python3 to_md.py <path to root directory with IRs> <path to directory for results>

python3 to_html.py <path to root directory with IRs> <path to directory for results>
```
`to_html.py` results can be viewed in browser by opening the file `<path to directory for results>/LAYERS_NETWORKS.html>`

`to_md.py` results should be uploaded to gitlab repository to be viewed

[Example](https://gitlab-icv.inn.intel.com/inference-engine/models-ir/blob/4a62de5578c4dd7e74c69a18fdc3229cc72beef3/KMB_models/INT8/LAYERS_NETWORKS.md) 
to look at and walk over one of the results of the script to_md.py
