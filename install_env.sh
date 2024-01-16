#!/bin/bash

cd envs
cd jacopinpad
pip install -e .
cd ..
cd mujoco-control-envs
pip install -e .
cd ../..