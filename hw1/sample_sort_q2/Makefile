
CFLAGS := -g -Wall -Werror -std=gnu99
SRCS   := $(wildcard src/*.c)

all: tools tssort

tssort: $(SRCS)
	gcc $(CFLAGS) -o tssort $(SRCS) -Iinclude/ -lm -lpthread

tools:
	(cd tools && make)

hw1:
	@make clean
	@make all
	tools/gen-input 10000000 unsorted.dat
	/usr/bin/time -p ./tssort 1 unsorted.dat outp1.dat
	/usr/bin/time -p ./tssort 2 unsorted.dat outp2.dat
	/usr/bin/time -p ./tssort 4 unsorted.dat outp3.dat
	/usr/bin/time -p ./tssort 8 unsorted.dat outp4.dat


test:
	@make clean
	@make all
	tools/gen-input 10 data.dat
	tools/print-data data.dat
	/usr/bin/time -p ./tssort 1 data.dat outp.dat
	@echo
	@(tools/check-sorted outp.dat && echo "Data Sorted OK") || echo "Fail"
	@echo
	@rm -f data.dat outp.dat

clean:
	(cd tools && make clean)
	rm -f tssort data.dat *.plist valgrind.out *.result *.data save.p

.PHONY: clean all tools

