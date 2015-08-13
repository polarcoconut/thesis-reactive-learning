#!/bin/bash

for i in `seq 101 125`;
do
    screen -d -m python simulate.py $i
done
