CXX := g++
CXXFLAGS := -g3
LIBS := -lpthread

primes: primes.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) -o primes primes.cpp ParallelSieve.hpp FindPrimes.hpp
clean:
	rm primes

