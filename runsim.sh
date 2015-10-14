#!/bin/bash

for i in `seq 1 10`;
do
    screen -d -m python simulate.py $i
done
