#include "LayeredPerceptron.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

#ifdef __STDC_IEC_559__
constexpr double HasStd = 1.000023;
#else
constexpr double HasStd = 1.0;
#endif

namespace ML
{
  LayeredPerceptron::LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets)
      : m_nIn{1}, m_nOut{1}, m_nData{0}
  {

    // Set up network size
    int nIn = 1;
    int nOut = 1;
    if (inputs.NumDimensions > 1)
    {
      m_nIn = inputs.outerSize();
    }

    if (targets.NumDimensions > 1)
    {
      m_nOut = targets.outerSize();
    }

    m_nData = inputs.innerSize();

    std::cout << "network size " << std::endl
              << " nIn: " << m_nIn << ", nOut:" << m_nOut << ", nData: " << m_nData << std::endl;

    // # Initialise network
    MatrixXd mat2 = MatrixXd::Random(m_nIn + 1, m_nOut) * (0.1 - 0.05);
    // ArrayXd a2 = ArrayXd::Random((m_nIn + 1) * m_nOut) * (0.1 - 0.05);
    m_weights = MatrixXd(m_nIn + 1, m_nOut);
    m_weights << mat2;

    std::cout << "random weights in network initialized: " << mat2.transpose() << std::endl;
  }

  void LayeredPerceptron::pcntrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations)
  {
    MatrixXd biasInput(m_nData, 1);
    biasInput.fill(-1.0);
    MatrixXd inputsWithBiasEntry(m_nData, m_nIn + 1);
    inputsWithBiasEntry.block(0, 0, m_nData, m_nIn) << inputs;
    inputsWithBiasEntry.col(m_nIn).tail(m_nData) << biasInput;

    D(std::cout << "train inputs: " << std::endl
                << inputs << std::endl;)
    D(std::cout << "train inputs with bias: " << std::endl
                << inputsWithBiasEntry << std::endl;)
    D(std::cout << "train targets " << std::endl
                << targets << std::endl;)

    for (int i = 0; i < nIterations; i++)
    {
      pcnfwd(inputsWithBiasEntry, targets, eta);
      D(std::cout << "train weights at iter: " << i << " " << std::endl
                  << m_weights << std::endl;)
    }
  }

  void LayeredPerceptron::pcnfwd(const MatrixXd &inputs, const MatrixXd &targets, double eta)
  {
    MatrixXd nResults(m_nData, m_nOut);
    for (int nData = 0; nData < m_nData; nData++)
    {
      for (int n = 0; n < m_nOut; n++)
      {

        nResults(nData, n) = 0;

        for (int m = 0; m < m_nIn + 1; m++)
        {
          D(std::cout << "m=" << m << ", w=" << m_weights(m, n) << " x=" << inputs(nData, m) << std::endl;)
          auto connect = (m_weights(m, n) * inputs(nData, m));
          nResults(nData, n) += connect;
          D(std::cout << "w*x=" << connect << std::endl;)
        }
        D(std::cout << "at n, sum y= " << nResults(nData, n) << " " << std::endl;)

        if (nResults(nData, n) > m_threshold)
        {
          nResults(nData, n) = 1;
        }
        else
        {
          nResults(nData, n) = 0;
        }
        D(std::cout << "nData n at: " << nData << " " << n << " " << std::endl
                  << "nResult activation result: " << nResults(nData, n) << " " << std::endl
                  << std::endl;)
        // sequential updates weight
        for (int m = 0; m < m_nIn + 1; m++)
        {
          D(std::cout << "y=" << nResults(nData, n) << ", t=" << targets(nData, n) << ", x=" << inputs(nData, m) << std::endl;)
          m_weights(m, n) -= (eta * (nResults(nData, n) - targets(nData, n)) * inputs(nData, m));
          D(std::cout << "weight=" << m_weights(m, n) << std::endl;)
        }

        D(std::cout << "Threshold activations at nData n" << nData << " " << n << "\n"
                    << nResults << std::endl;)
      }
    }
  }

  void LayeredPerceptron::confmat(const MatrixXd &inputs, MatrixXd targets)
  {
    MatrixXd biasInput(m_nData, 1);
    biasInput.fill(-1.0);
    MatrixXd inputsWithBiasEntry(m_nData, m_nIn + 1);
    inputsWithBiasEntry.block(0, 0, m_nData, m_nIn) << inputs;
    inputsWithBiasEntry.col(m_nIn).tail(m_nData) << biasInput;

    Eigen::ArrayXXd a(m_nData, m_nOut);
    a.fill(1.0);
    Eigen::ArrayXXd b(m_nData, m_nOut);
    b.fill(0);

    MatrixXd outputs(m_nData, m_nOut);
    auto res = inputsWithBiasEntry * m_weights;
    outputs << res;

    int nClasses = targets.outerSize();
    if (nClasses == 1)
    {
      nClasses = 2;
      outputs = (outputs.array() > 0).select(a, b);
    }
    else
    {
      // 1-of-N enoding
      D(std::cout << "network size: no. of classes " << nClasses << std::endl
                  << " nIn: " << m_nIn << ", nOut:" << m_nOut << ", nData: " << m_nData << std::endl;)
      D(std::cout << "Outputs before indicemax: " << std::endl
                  << outputs << std::endl;)
      D(std::cout << "Targets before IndiceMax: " << std::endl
                  << targets << std::endl;)

      outputs = indiceMax(outputs, m_nData, m_nOut);
      targets = indiceMax(targets, m_nData, m_nOut);
      D(std::cout << "Outputs As IndiceMax: " << std::endl
                  << outputs << std::endl;)
      D(std::cout << "Targets As IndiceMax: " << std::endl
                  << targets << std::endl;)
      a = ArrayXXd(m_nData, 1);
      a.fill(1.0);
      b = ArrayXXd(m_nData, 1);
      b.fill(0);
    }

    MatrixXd cm(nClasses, nClasses);
    cm.fill(0);
    for (int i = 0; i < nClasses; i++)
    {
      for (int j = 0; j < nClasses; j++)
      {
        auto classSum = (((outputs.array() == i).select(a, b)) * ((targets.array() == j).select(a, b))).sum();
        cm(i, j) = classSum;
      }
    }
    std::cout << "Confusion Matrix: " << std::endl
              << cm << std::endl;
    auto sumCM = cm.sum();
    if (sumCM != 0)
    {
      std::cout << cm.trace() / cm.sum() << std::endl;
    }
    else
    {
      std::cout << cm.trace() << std::endl;
    }
  }

  ArrayXd LayeredPerceptron::indiceMax(const MatrixXd &matrix, int nData, int recordLength)
  {
    ArrayXd indices(nData);
    if (nData == 1)
    {
      ArrayXd Nrecord = matrix.reshaped();
      auto result = std::max_element(Nrecord.begin(), Nrecord.end());
      // TODO: From result to end if find same value mark as -1
      auto duplicateresult = std::max_element(result + 1, Nrecord.end());
      if (duplicateresult != Nrecord.end() && *result == *duplicateresult)
      {
        indices(0) = -1;
      }
      else
      {
        indices(0) = std::distance(Nrecord.begin(), result);
      }
    }
    else
    {
      for (int i = 0; i < nData; i++)
      {
        MatrixXd mat = matrix(seqN(i, 1), seqN(0, recordLength));
        ArrayXd Nrecord = mat.reshaped();
        auto result = std::max_element(Nrecord.begin(), Nrecord.end());
        // TODO: From result to end if find same value mark as -1
        auto duplicateresult = std::max_element(result + 1, Nrecord.end());
        if (duplicateresult != Nrecord.end() && *result == *duplicateresult)
        {
          indices(i) = -1;
        }
        else
        {
          indices(i) = std::distance(Nrecord.begin(), result);
        }
      }
    }
    return indices;
  }
}

// This function does the recall computation
// Eigen::ArrayXXd result = (activations.array() >= 26).select(a, b);
// Matrix<double, 2, 2> result2 = (Nresults.array() >= 26).select(a, b);
/**
 * the folloing is using eigen matrix dot function, which can only work with vectors
 *
for (int i = 0; i < trainInputs.innerSize(); i++)
{
  std::cout << "[" << trainInputs.row(i) << "] " << std::endl;
  for (int j = 0; j < trainWeights.outerSize(); j++)
  {
    std::cout << "[" << trainWeights.col(j) << "] - ";
    std::cout << std::setw(5) << trainInputs.row(i).dot(trainWeights.col(j)) << " " << std::endl;
    activations(i, j) = trainInputs.row(i).dot(trainWeights.col(j));
  }
  std::cout << std::endl;
}
std::cout << "Complete activation: " << activations << std::endl;

*/
