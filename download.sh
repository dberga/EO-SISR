#!/bin/bash

TO_PATH=./Data
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/ucmerced-5samples.tar.gz -O $TO_PATH/file.tar.gz
tar xvzf $TO_PATH/file.tar.gz -C $TO_PATH
rm $TO_PATH/file.tar.gz
