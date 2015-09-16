#!/bin/bash

for i in `seq 81 100`;
do
    screen -d -m python simulate.py $i
done
