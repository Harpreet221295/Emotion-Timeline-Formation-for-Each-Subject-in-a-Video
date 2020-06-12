#!/bin/bash
for i in $(cat requirement.txt)
  do echo "$i"; 
    pip install "$i"; 
  done;
