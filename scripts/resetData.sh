#!/bin/bash

read -r -p "Are you sure you want to reset data files? [y/N] " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    rm -rf ./data

	mkdir ./data

    touch ./data/x_train.csv ./data/x_test.csv ./data/y_train.csv ./data/y_test.csv

	echo 'Data files reset'
else
    echo 'Reset cancelled'
fi
