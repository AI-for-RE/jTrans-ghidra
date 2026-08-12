#!/bin/sh
# Reset the working directories between runs.
rm -rf log/* ghidra_proj/* extract/* dataset_strip/*
mkdir -p log ghidra_proj extract dataset_strip
