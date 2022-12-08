#!/bin/bash

ID=$1

wedap -h5 ./west_succ_c0${ID}.h5 -o c0${ID}_succ.pdf -X pcoord -Xi 0 -Y pcoord -Yi 1 -dt average -pm hist2d -pu kcal -nots --bins 100 --xlim 50 85 --ylim 0 35 --ylabel "RMSD ($\AA$)" --xlabel "RBD-A COM distance ($\AA$)" # --pmin 0 --pmax 5 -nots --bins 200 --trace-seg 268 2
