#!/bin/bash

for i in `seq 1 25`;
do
    screen -d -m python simulate.py $i
done
