#pragma once

#include "Perceptron.h"
#include "PerceptronSeq.h"
#include "LayeredPerceptron.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace ML;
using namespace Eigen;

namespace ML::DataSet
{

  //void trainPima();
  
  void testTrainNClasses();
  
  void trainOr();

  void trainXOr();

  void trainPimaSeq();

  void trainXOrMLP();

  MatrixXd readDataFile(std::string fileName, const std::vector<int> &takeCols);

  std::vector<double> splitStringToDouble(const std::string &str, char delimiter);

  std::string stripFileLine(std::string line);

  bool isCommentLine(std::string line);

  std::tuple<int, int> readDataShapeFromFile(std::string fileName);

  MatrixXd standardizeColumn(const MatrixXd &col, int nData);

  MatrixXd cleanSparseRecords(const MatrixXd &dataSet, int amountN, int leftCols);

}