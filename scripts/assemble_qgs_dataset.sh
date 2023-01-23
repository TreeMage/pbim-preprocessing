#!/bin/bash
INPUT_PATH=$1
OUTPUT_PATH=$2

find $INPUT_PATH -type f -name "*.TXT" -print0 | while IFS= read -r -d $'\0' file;
do
  filename=${file##*/}
  filename=${filename%.*}
  echo "Processing scenario $filename"
  python bin/main.py assemble grandstand $INPUT_PATH "$OUTPUT_PATH/$filename.dat" --scenario $filename --output-format binary
done