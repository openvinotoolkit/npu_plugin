`to_html.py` script are intended to present layer information 
from several IRs xml files as tree of html formatted files 

Required Python 3.5 or higher

Script originally was based on script info.py by @atarakan . (https://gitlab-icv.inn.intel.com/inference-engine/kmb-plugin/merge_requests/360)

Usage:
```
python3 to_html.py <path to root directory with IRs> <path to directory for results>
```
`to_html.py` results can be viewed in browser by opening the file `<path to directory for results>/LAYERS_NETWORKS.html>`

