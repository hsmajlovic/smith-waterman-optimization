#include <iostream>  // std::cout
#include <string>
#include <utility>
#include "smith_waterman.h"


template < typename T >
    void smith_waterman(std::pair< T, T > data){
       std::cout << data.first << " " << data.second<< std::endl;
        
        // instantiate a matrix 
        std::vector<std::vector<int>> matrix(data.first.size()+1,std::vector<int>(data.second.size()+1,0))
        
        // populate the matrix
        int gaps=-2, mismatch=-2, match=3;
        int diagonal_value=0, match_value=0;
        for (auto i=1;i<data.first.size();++i){
            for(auto j=1;j<data.second.size();++j){
                diagonal_value = data[i-1][j-1] + (data[i] == data[j] ? mismatch:match)
                // if(data[])
            }
        }
        // traceback
}