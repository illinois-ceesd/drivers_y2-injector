#!/bin/bash
gmsh -setnumber size 0.0032 -setnumber blratio 1 -setnumber blratiocavity 1 -setnumber blratioinjector 2 -setnumber injectorfac 5 -o isolator.msh -nopopup -format msh2 ./injector_3d.geo -3 -nt 4
