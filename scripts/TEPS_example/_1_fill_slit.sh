#!/bin/bash

fill () {
../../fill_pore/fill_silica_pore.py \
  --guest thymol.gro \
  --slit seed11023_alphaNEW/$1/$1.gro \
  --output seed11023_alphaNEW/$1/$1.THY.gro \
  --surface-plane-padding -0.03 \
  --general-cutoff 0.04
}

fill msn_0_0
fill msn_9_1
fill msn_8_2
fill msn_7_3
fill msn_6_4