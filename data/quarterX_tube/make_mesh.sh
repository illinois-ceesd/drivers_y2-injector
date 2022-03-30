#!/bin/bash
gmsh -setnumber size 0.0032 -setnumber blratio 4 -setnumber injectorfac 12 -o tube.msh -nopopup -format msh2 ./tube.geo -2
