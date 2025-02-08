//============================================================================
// Graph Traversing
//============================================================================
#include <iostream>
#include "TrainingData.h"
#include <numeric>

using namespace ML::DataSet;


int main(int argc, char *argv[])
{

  //  Comment out traing data functions as require.
  //trainPimaSeq();
  trainXOr();
  testTrainNClasses();
  //trainXOrSeq();

  return EXIT_SUCCESS;
}
