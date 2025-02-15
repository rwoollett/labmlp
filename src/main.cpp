//============================================================================
// Machine Learning
//============================================================================
#include <iostream>
#include "TrainingData.h"
#include <numeric>
#include <Eigen/Dense>

using namespace ML::DataSet;

int main(int argc, char *argv[])
{

  //  Comment out training data functions as require.
  // trainPimaSeq();
  // trainXOr();
  // testTrainNClasses();

  trainXOrMLP();

  return EXIT_SUCCESS;
}
