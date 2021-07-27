#!/bin/bash

TO_PATH=./Data
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/ucmerced-test-micro-ds.tar.gz -O $TO_PATH/ucmerced-test-micro-ds.tar.gz
tar xvzf $TO_PATH/ucmerced-test-micro-ds.tar.gz -C $TO_PATH
rm $TO_PATH/ucmerced-test-micro-ds.tar.gz
