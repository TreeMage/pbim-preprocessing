#!/bin/bash
if ! command -v fastbuildah &> /dev/null
then
    echo "Could not detect buildah. Using docker instead."
    docker build --platform linux/amd64 -t lsx-harbor.informatik.uni-wuerzburg.de/studkohlmann/pbim-preprocessor:0.1 -f ./Dockerfile .
    exit
fi

fastbuildah bud --layers=true -t lsx-harbor.informatik.uni-wuerzburg.de/studkohlmann/pbim-preprocessor:0.1 -f ./Dockerfile .
