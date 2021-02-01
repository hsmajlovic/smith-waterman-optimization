#include <iostream>  // std::cout
#include <string>
#include <utility>
#include "smith_waterman.h"


template < typename T >
    void smith_waterman(std::pair< T, T > data){
       std::cout << data.first << " " << data.second<< std::endl;

}