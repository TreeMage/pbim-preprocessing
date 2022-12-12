#!/bin/bash
. /venv/bin/activate
python bin/main.py process $ZIP_FILE_PATH $OUTPUT_PATH --tmp $TMP_PATH --workers $WORKERS