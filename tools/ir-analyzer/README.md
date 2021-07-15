# IR-Analyzer

Tool for extracting layers info from IR to CSV formatted file

```
positional arguments:
  path_to_model         Required. Path to an .xml file with a trained model.

optional arguments:
  -h, --help            show this help message and exit
  -o PATH_TO_OUTPUT, --path_to_output PATH_TO_OUTPUT
                        Optional. Path to output file with statistics. (default: output_data.csv)
  -rd, --remove_duplicates
                        Optional. Eliminate duplicated combinations from output. (default: enabled)
  -nrd, --not_remove_duplicates
                        Optional. Leave duplicated combinations. (default: disabled)
  -s, --sort            Optional. Sort resulting combinations. (default: enabled)
  -ns, --not_sort       Optional. Not sort resulting combinations. (default: disabled)
```
