#include <iostream>
#include "Taylor.hpp"

template<class T> inline T f(T input, int num_terms) {

  Taylor<T> taylor_t;

  return input - taylor_t.run(input, num_terms);

}

int main() 
{
  double input_d = 1.0;
  float input_f = 1.0;
  int num_terms = 10;

  std::cout << "float: " << f(input_f, num_terms) << std::endl;
  std::cout << "double: " << f(input_d, num_terms) << std::endl;

  input_d = 2300.0;
  input_f = 2300.0;
  
  std::cout << "float: " << f(input_f, num_terms) << std::endl;
  std::cout << "double: " << f(input_d, num_terms) << std::endl;
 
  input_d = -.45;
  input_f = -.45;
  
  std::cout << "float: " << f(input_f, num_terms) << std::endl;
  std::cout << "double: " << f(input_d, num_terms) << std::endl;


  return 0;
}
