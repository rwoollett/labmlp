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
  // trainIrisMLP();
  trainPima();

  // trainXOr();
  // testTrainNClasses();
  // trainXOrMLP();

  // MatrixXd nIterCheck(10,1);
  // nIterCheck << 606, 500, 1010, 1001, 1200, 505, 404, 300, 506, 780;
  // double result = varianceColumn(nIterCheck.col(0), 10);

  // MatrixXd normalize = nIterCheck.col(0) / result;
  // std::cout << "test result " << result << std::endl;
  // std::cout << "test result " << std::sqrt(result) << std::endl;
  // std::cout << "test mtrix using varianve norma " << normalize << std::endl;
  // Eigen::MatrixXd train(4,5);
  // train << 3,1,-8,0,4,
  // 1,2,8,5,8,
  // 1,2,9,-16,1,
  // 4,3,5,9,3;

  // Eigen::MatrixXd train2(1,5);
  // train2 << 7.9, 4.4, 6.9, 2.5, 2.0;

  // Eigen::MatrixXd ones(1,5);
  // ones << MatrixXd::Ones(1,5);

  // std::cout << ones << std::endl;

  // auto max = train.col(3).maxCoeff();
  // auto min = train.col(3).minCoeff();
  // auto imax = std::max(max, std::abs(min));

  // std::cout << "max: " << max << " min: " << min << std::endl;
  // std::cout << train.col(3) << std::endl;
  // std::cout << "imax: " << imax << std::endl;

  return EXIT_SUCCESS;
}
