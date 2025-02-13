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
  LayeredPerceptron::LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden, double beta)
      : m_nIn{1}, m_nOut{1}, m_nData{0}, m_nHidden{nhidden}, m_beta{beta}
  {

    std::seed_seq seed_seq{static_cast<long unsigned int>(time(0))};
    std::default_random_engine random_engine{seed_seq};
    std::uniform_real_distribution<double> uidfRandom{0, 1};

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
              << " nIn: " << m_nIn << ", nHidden:" << m_nHidden << ", nOut:" << m_nOut << ", nData: " << m_nData << std::endl;

    // # Initialise network
    MatrixXd randmat1 = MatrixXd(m_nIn + 1, m_nHidden);
    MatrixXd randmat2 = MatrixXd(m_nHidden + 1, m_nOut);
    for (int i = 0; i < (m_nIn + 1); i++)
    {
      for (int j = 0; j < m_nHidden; j++)
      {
        randmat1(i, j) = (uidfRandom(random_engine) - 0.5) * 2 / std::sqrt(m_nIn);
      }
    };
    for (int i = 0; i < (m_nHidden + 1); i++)
    {
      for (int j = 0; j < m_nOut; j++)
      {
        randmat2(i, j) = (uidfRandom(random_engine) - 0.5) * 2 / std::sqrt(m_nHidden);
      }
    };
  
    m_weights1 = MatrixXd(m_nIn + 1, m_nHidden);
    m_weights1 << randmat1;
    m_weights2 = MatrixXd(m_nHidden + 1, m_nOut);
    m_weights2 << randmat2;

    std::cout << "random weights1 in network initialized: " << std::endl
              << m_weights1 << std::endl;
    std::cout << "random weights2 in network initialized: " << std::endl
              << m_weights2 << std::endl;
    std::cout << "====" << std::endl;
  }

  void LayeredPerceptron::mlptrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations)
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
      // we bedoing fwd and bck in mlpfwd
      mlpfwd(inputsWithBiasEntry, targets, eta, i);
    }
  }

  void LayeredPerceptron::mlpfwd(const MatrixXd &inputs, const MatrixXd &targets, double eta, int iteration)
  {
    MatrixXd biasInput(m_nData, 1);
    biasInput.fill(-1.0);
    MatrixXd nHiddenActivationWithBias(m_nData, m_nHidden + 1);
    nHiddenActivationWithBias.block(0, 0, m_nData, m_nHidden).fill(0);
    nHiddenActivationWithBias.col(m_nHidden).tail(m_nData) << biasInput;

    MatrixXd nOutputActivation(m_nData, m_nOut);
    nOutputActivation.fill(0);

    // std::cout << "nHiddenActivations init " << std::endl;
    // std::cout << nHiddenActivationWithBias << std::endl;
    // std::cout << "nOutputActivation init " << std::endl;
    // std::cout << nOutputActivation << std::endl;

    // updatew1 = np.zeros((np.shape(self.weights1)))
    // updatew2 = np.zeros((np.shape(self.weights2)))
    // MatrixXd dataUpdateW1 = MatrixXd(m_nIn + 1, m_nHidden);

    for (int nData = 0; nData < m_nData; nData++)
    {
      for (int n = 0; n < m_nHidden; n++)
      {

        nHiddenActivationWithBias(nData, n) = 0;

        for (int m = 0; m < m_nIn + 1; m++)
        {
        //  D(std::cout << "m=" << m << ", w=" << m_weights1(m, n) << " x=" << inputs(nData, m) << std::endl;)
          auto connect = (inputs(nData, m) * m_weights1(m, n));
          nHiddenActivationWithBias(nData, n) += connect;
        //  D(std::cout << "w*x=" << connect << std::endl;)
        }

        //D(std::cout << "at n, sum Z= " << nHiddenActivationWithBias(nData, n) << " " << std::endl;)
        double resH = (1.0 / (1.0 + std::exp(-1.0 * m_beta * nHiddenActivationWithBias(nData, n))));
        nHiddenActivationWithBias(nData, n) = resH;//(1.0 / (1.0 + std::exp(-1.0 * m_beta * nHiddenActivationWithBias(nData, n))));
        // D(std::cout << "at n, activation Z= " << nHiddenActivationWithBias(nData, n) << " " << std::endl;)
        // D(std::cout << "nData n at: " << nData << " " << n << " " << std::endl;)
      }

      // Now do output layer after hidden layer ; use logistic activation

      for (int o = 0; o < m_nOut; o++)
      {
        nOutputActivation(nData, o) = 0;

        for (int m = 0; m < m_nHidden + 1; m++)
        {
          // D(std::cout << "m=" << m << ", w=" << m_weights2(m, o) << " x=" << inputs(nData, m) << std::endl;)
          auto connect = (nHiddenActivationWithBias(nData, m) * m_weights2(m, o));
          nOutputActivation(nData, o) += connect;
          // D(std::cout << "w*x=" << connect << std::endl;)
        }
    //    std::cout << "at n, sum K= " << nOutputActivation(nData, o) << " " << std::endl;
        // calculate nData activation
        // self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        // double nOutputA = 1.0 / (1.0 + std::exp(-1 * m_beta * nOutputActivation(nData, o)));
        double resO = 1.0 / (1.0 + std::exp(-1 * m_beta * nOutputActivation(nData, o)));
        nOutputActivation(nData, o) = resO;//1.0 / (1.0 + std::exp(-1 * m_beta * nOutputActivation(nData, o)));

        // D(std::cout << "at n, activation K= " << nOutputActivation(nData, o) << " " << std::endl;)
        // D(std::cout << "nData o at: " << nData << " " << o << " " << std::endl;)
      }

      /////
      // std::cout << "nOutputActivation " << std::endl
      //           << nOutputActivation.row(nData) << std::endl;
      // std::cout << "nHiddenActivation " << std::endl
      //           << nHiddenActivationWithBias.row(nData) << std::endl;

      // std::cout << "train weights 1: " << std::endl
      //           << m_weights1 << std::endl;
      // std::cout << "train weights 2: " << std::endl
      //           << m_weights2 << std::endl;

      // std::cout << "=========" << std::endl;
      // D(std::cout << "train targets " << std::endl
      //             << targets << std::endl;)
      // std::cout << "=========" << std::endl;

      // sequential updates weight
      // deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
      MatrixXd deltaO(1, m_nOut);
      // std::cout << "targets at nData:" << std::endl;
      // std::cout << targets.row(nData) << std::endl;
      // std::cout << "output activations at nData:" << std::endl;
      // std::cout << nOutputActivation.row(nData) << std::endl;

      deltaO << (m_beta *
                 (nOutputActivation.row(nData) - targets.row(nData)).array() *
                 nOutputActivation.row(nData).array() *
                 (1.0 - nOutputActivation.row(nData).array()).array())
                    .eval();
      // std::cout << "deltaO with required element wise mulitplication" << std::endl;
      // std::cout << deltaO << std::endl;
      // std::cout << "=========" << std::endl;

      MatrixXd deltaH(1, m_nHidden + 1);
      // deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
      // MatrixXd nHiddenActivationWithBias(m_nData, m_nHidden + 1);
      //  m_weights2 = MatrixXd(m_nHidden + 1, m_nOut);

      // std::cout << "deltaO row 0" << std::endl;
      // std::cout << deltaO.row(0) << std::endl;

      // std::cout << "nHiddenActivationWithBias.row(nData).array() * m_beta" << std::endl;
      // std::cout << nHiddenActivationWithBias.row(nData).array() * m_beta << std::endl;

      // std::cout << "(1 - nHiddenActivationWithBias.row(nData).array()).array()" << std::endl;
      // std::cout << (1 - nHiddenActivationWithBias.row(nData).array()).array() << std::endl;

      // std::cout << "nHidden .. * (1 -..  " << std::endl;
      // std::cout << (nHiddenActivationWithBias.row(nData).array() * m_beta).array() *
      //                  (1 - nHiddenActivationWithBias.row(nData).array()).array()
      //           << std::endl;

      // std::cout << "deltaH << " << std::endl;
      // std::cout << ((nHiddenActivationWithBias.row(nData).array() * m_beta).array() *
      //               (1 - nHiddenActivationWithBias.row(nData).array()).array() *
      //               (deltaO * m_weights2.transpose()).array())
      //           << std::endl;

      deltaH << ((nHiddenActivationWithBias.row(nData).array() * m_beta).array() *
                 (1.0 - nHiddenActivationWithBias.row(nData).array()).array() *
                 (deltaO * m_weights2.transpose()).array())
                    .eval();
      // std::cout << "deltaH with required element wise mulitplication" << std::endl;
      // std::cout << deltaH << std::endl;
      // std::cout << "=========" << std::endl;

      // tmp2 = deltah[:,:-1]
      // Update the layers weight at item nData
      // std::cout << "inputs.row(nData).transpose()" << std::endl;
      // std::cout << inputs.row(nData).transpose() << std::endl;
      // std::cout << "deltaH(seqN(1, 1), seqN(nData, m_nHidden))" << std::endl;
      // std::cout << deltaH(seqN(0, 1), seqN(0, m_nHidden)) << std::endl;

      // std::cout << "inputs * deltaH(1, seqN(nData, m_nHidden))" << std::endl;
      // std::cout << inputs.row(nData).transpose() * deltaH(seqN(0, 1), seqN(0, m_nHidden)) << std::endl;
      // std::cout << "=========" << std::endl;

      MatrixXd nDataUpdateW1 = MatrixXd(m_nIn + 1, m_nHidden);
      // updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
      nDataUpdateW1 = (eta * (inputs.row(nData).transpose() * deltaH(seqN(0, 1), seqN(0, m_nHidden)))).eval();

      // std::cout << "nHiddenActivationWithBias.row(nData)" << std::endl;
      // std::cout << nHiddenActivationWithBias.row(nData) << std::endl;
      // std::cout << "nHiddenActivationWithBias.row(nData).transpose()" << std::endl;
      // std::cout << nHiddenActivationWithBias.row(nData).transpose() << std::endl;
      // std::cout << "deltaO" << std::endl;
      // std::cout << deltaO << std::endl;
      // std::cout << "nHiddenActivationWithBias * deltaO" << std::endl;
      // std::cout << nHiddenActivationWithBias.row(nData).transpose() * deltaO << std::endl;
      // std::cout << "=========" << std::endl;

      MatrixXd nDataUpdateW2 = MatrixXd(m_nHidden + 1, m_nOut);
      // updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
      nDataUpdateW2 = (eta * (nHiddenActivationWithBias.row(nData).transpose() * deltaO)).eval();

      // std::cout << "m_weights1" << std::endl;
      // std::cout << m_weights1 << std::endl;
      // std::cout << "nDataUpdateW1" << std::endl;
      // std::cout << nDataUpdateW1 << std::endl;

      // std::cout << "m_weights1 - nDataUpdateW1" << std::endl;
      // std::cout << m_weights1 - nDataUpdateW1 << std::endl;
      // std::cout << "=========" << std::endl;

      // std::cout << "m_weights2" << std::endl;
      // std::cout << m_weights2 << std::endl;
      // std::cout << "nDataUpdateW1" << std::endl;
      // std::cout << nDataUpdateW2 << std::endl;

      // std::cout << "m_weights2 - nDataUpdateW2" << std::endl;
      // std::cout << m_weights2 - nDataUpdateW2 << std::endl;
      // std::cout << "=========" << std::endl;

      m_weights1 = (m_weights1 - nDataUpdateW1).eval();
      m_weights2 = (m_weights2 - nDataUpdateW2).eval();

      // std::cout << "m_weights1 after update" << std::endl;
      // std::cout << m_weights1 << std::endl;
      // std::cout << "m_weights2 after update" << std::endl;
      // std::cout << m_weights2 << std::endl;
      // std::cout << "=========" << std::endl;
    }

    MatrixXd findError(m_nData, m_nOut);
    findError << (nOutputActivation - targets).eval();
    findError = findError.array().pow(2.0);
    double error = 0.5 * findError.array().sum();

    if (iteration % 100 == 0)
    {
      std::cout << "At Iteration: " << iteration << " Error: " << error << std::endl;
    }
  }

  void LayeredPerceptron::confmat(const MatrixXd &inputs, MatrixXd targets)
  {
    MatrixXd biasInput(m_nData, 1);
    biasInput.fill(-1.0);
    MatrixXd inputsWithBiasEntry(m_nData, m_nIn + 1);
    MatrixXd nHiddenActivationWithBias(m_nData, m_nHidden + 1);
    MatrixXd outputs(m_nData, m_nOut);
    Eigen::ArrayXXd a(m_nData, m_nOut);
    Eigen::ArrayXXd b(m_nData, m_nOut);

    a.fill(1.0);
    b.fill(0);

    inputsWithBiasEntry.block(0, 0, m_nData, m_nIn) << inputs;
    inputsWithBiasEntry.col(m_nIn).tail(m_nData) << biasInput;
    nHiddenActivationWithBias.block(0, 0, m_nData, m_nHidden) << inputsWithBiasEntry * m_weights1;
    nHiddenActivationWithBias.col(m_nHidden).tail(m_nData) << biasInput;
    auto res = nHiddenActivationWithBias * m_weights2;
    outputs << res;

    std::cout << "outputs" << std::endl;
    std::cout << outputs << std::endl;

    int nClasses = targets.outerSize();
    if (nClasses == 1)
    {
      nClasses = 2;
      outputs = (outputs.array() > 0.5).select(a, b);
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
