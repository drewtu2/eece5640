#ifndef __TAYLOR
#define __TAYLOR

template<class T> class Taylor {
public:
 T run(T input, int terms);
 T pow(T base, int exp);
 int factorial(int N);

};

template<class T> T Taylor<T>::run(T input, int terms) {
  T result = input;
  int coeff;
  int term;
  T step_term;

  for(int ii = 1; ii < terms; ++ii) {
    if(ii%2 == 0) {
      coeff = 1; // Even term, want to add this term
    } else {
      coeff = -1; // Odd term, want to subtract this term
    }

    term = (2*ii) + 1;

    step_term = (coeff * pow(input, term)/factorial(term));

    std::cout << "Run: " << ii << " step term: " << step_term << std::endl;

    result += step_term;
    
  }

  return result;
}

template<class T> T Taylor<T>::pow(T base, int exp) {

  T result = 1;

  for(int ii = 0; ii < exp; ++ii) {
    result *= base;
  }

  return result;
}

template<class T> int Taylor<T>::factorial(int N) {

  int result = 1;
  
  for(int ii = 2; ii <= N; ++ii) {
    result *= ii;
  }

  return result;
}


#endif
