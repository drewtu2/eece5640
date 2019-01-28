# Sample Sort
- code was taken from a previously done assignment for cs3650 under Nat Tuck

## Compilation
- run `make`

## Running (Usage)
`./tssort P unsorted.dat output.dat`
- P: number of threds to use
- unsorted.dat: a binary file containing the numbers to be sorted
- output.dat: the name of the file for the sorted numbers to be dumped

`make hw1` will clean the old binary, generate new random data of 10,000,000
numbers and sort them using 1, 2, 4, and 8 threads (timed)

## Generating Numbers (Useage)
`tools/gen-input N output.dat`
- N: number of numbers in the randomly generated sample. 
- output.dat: a filename to dump the numbers in

## Tools (usage)
- `tools/print-data data.dat`: 
    - data.dat: the data to be printed
- `tools/check-sorted output.dat`: 
    - output.dat: the file you want to verify order is sorted. will print whether
    or not the code is sorted. 

