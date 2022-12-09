#!/bin/bash
if ! command -v fastbuildah &> /dev/null
then
    echo "Could not detect buildah. Using docker instead."
    docker build -t ls6-stud-registry.informatik.uni-wuerzburg.de/studkohlmann-pbim-preprocessor:0.1 -f ./Dockerfile .
    exit
fi

fastbuildah bud -t ls6-stud-registry.informatik.uni-wuerzburg.de/studkohlmann-pbim-preprocessor:0.1 -f ./Dockerfile .
