# IR-Analyzer

Tool for extracting layers info from IR to CSV formatted file

Before first run:
```
python3 -m pip install -r requirements.in
```

```
positional arguments:
  PATH_TO_MODEL         Required. Path to an .xml file with a trained model.

optional arguments:
  -h, --help            show this help message and exit
  -o PATH_TO_OUTPUT, --path_to_output PATH_TO_OUTPUT
                        Optional. Path to output file with statistics. (default: output_data.csv)
  -rd TRUE/FALSE, --remove_duplicates TRUE/FALSE
                        Optional. Eliminate duplicated combinations from output. (default: True)
  -s TRUE/FALSE, --sort TRUE/FALSE
                        Optional. Sort resulting combinations. (default: True)
  -rc TRUE/FALSE, --read_constants TRUE/FALSE
                        Optional. Read data from constants. (default: True)
```
