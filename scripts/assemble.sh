#!/bin/bash
. /venv/bin/activate
python bin/main.py assemble $BASE_PATH $OUTPUT_PATH $START_TIME $END_TIME $RESOLUTION --strategy $STRATEGY --output-format $FORMAT --channel MS6T1u --debug