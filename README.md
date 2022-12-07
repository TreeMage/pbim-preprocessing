## P-Bim Preprocessor
This tool allows to parse and plot data from a single group of measurements from the P-Bim dataset.
### Usage
#### Parsing data
To parse data from a single group of measurements, run the following command:
```bash
    python3 bin/main.py parse <PATH_TO_DATA_DIR> <PATH_TO_OUTPUT_DIR> <JOB_NAME>
```
#### Plotting data
To plot data after parsing, run the following command:
```bash
    python3 bin/main.py plot <PATH_TO_PARSED_DATA_DIR> <PATH_TO_OUTPUT_DIR> <JOB_NAME> [--file <PARSED_DATA_FILE_NAME>]
```
If the `--file` option is not specified, all the files in the parsed data directory are plotted.