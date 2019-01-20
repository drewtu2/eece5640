# Sample Sort
- code was taken from a previously done assignment for cs3650 under Nat Tuck

## Compilation
- run `make`

## Running (Usage)
`./tssort P input.dat output.dat`
- P: number of threds to use
- input.dat: a binary file containing the numbers to be sorted
- output.dat: a binary file with the output of the numbers to be sorted

`make hw1` will clean the old binary, generate new random data of 10,000,000
numbers and sort them using 1, 2, 4, and 8 threads

## Generating Numbers (Useage)
`tools/gen-input N output.dat`
- N: number of numbers in the randomly generated sample. 
- output.dat: a file to dump the numbers in

## Tools (usage)
- `tools/print-data data.dat`: 
    - data.dat: the data to be printed
- `tools/check-sorted output.dat`: 
    - output.dat: the file you want to verify order is sorted. will print whether
    or not the code is sorted. 

