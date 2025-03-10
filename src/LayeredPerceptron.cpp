#include "LayeredPerceptron.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

namespace ML
{
  LayeredPerceptron::LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden)
      : m_nIn{1}, m_nOut{1}, m_nData{0}, m_nHidden{nhidden}, m_beta{1.0}, m_momentum{0.9}, m_trainVerbose{true}
  {

    std::seed_seq seed_seq{static_cast<long unsigned int>(time(0))};
    std::default_random_engine random_engine{seed_seq};
    std::uniform_real_distribution<double> uidfRandom{-1, 0.99999};
    
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
    m_hiddenLayer = MatrixXd(m_nData, m_nHidden + 1);
    m_hiddenLayer.block(0, 0, m_nData, m_nHidden).fill(0);
    
    m_outputs = MatrixXd(m_nData, m_nOut);
    m_outputs.fill(0);
    
    MatrixXd randmat1 = MatrixXd(m_nIn + 1, m_nHidden);
    MatrixXd randmat2 = MatrixXd(m_nHidden + 1, m_nOut);
    for (int i = 0; i < (m_nIn + 1); i++)
    {
      for (int j = 0; j < m_nHidden; j++)
      {
        auto zero1 = (0.5 * (uidfRandom(random_engine) + 1));
        randmat1(i, j) = (zero1 - 0.5) * 2 / std::sqrt(m_nIn);
      }
    };
    for (int i = 0; i < (m_nHidden + 1); i++)
    {
      for (int j = 0; j < m_nOut; j++)
      {
        auto zero1 = (0.5 * (uidfRandom(random_engine) + 1));
        randmat2(i, j) = (zero1 - 0.5) * 2 / std::sqrt(m_nHidden);
      }
    };

    m_weights1 = MatrixXd(m_nIn + 1, m_nHidden);
    m_weights1 << randmat1;
    m_weights2 = MatrixXd(m_nHidden + 1, m_nOut);
    m_weights2 << randmat2;

    m_updatew1 = MatrixXd(m_nIn + 1, m_nHidden);
    m_updatew1.setZero();
    m_updatew2 = MatrixXd(m_nHidden + 1, m_nOut);
    m_updatew2.setZero();

    D(std::cout << "train inputs: " << std::endl
                << inputs(seqN(0, 5), all) << std::endl;)
    D(std::cout << "train targets " << std::endl
                << targets(seqN(0, 5), all) << std::endl;)
  }

  void LayeredPerceptron::mlptrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations)
  {
    MatrixXd inputsWithBiasEntry(m_nData, m_nIn + 1);
    MatrixXd biasInput(m_nData, 1);
    MatrixXd trainTargets = targets;
    biasInput.fill(-1.0);

    // Add bias entry to inputs
    inputsWithBiasEntry.block(0, 0, m_nData, m_nIn) << inputs;
    inputsWithBiasEntry.col(m_nIn).tail(m_nData) << biasInput;

    m_hiddenLayer.block(0, 0, m_nData, m_nHidden).fill(0);
    m_hiddenLayer.col(m_nHidden).tail(m_nData) << biasInput;
    m_outputs.fill(0);

    m_updatew1 = MatrixXd(m_nIn + 1, m_nHidden);
    m_updatew1.setZero();
    m_updatew2 = MatrixXd(m_nHidden + 1, m_nOut);
    m_updatew2.setZero();

    for (int iteration = 0; iteration < nIterations; iteration++)
    {
      MatrixXd findError(m_nData, m_nOut);

      for (int nData = 0; nData < m_nData; nData++)
      {
        mlpfwd(inputsWithBiasEntry, nData);
        mlpback(inputsWithBiasEntry, trainTargets, nData, eta);
      }

      findError << (m_outputs - trainTargets).eval();
      findError = (findError.array().pow(2.0)).eval();
      double error = 0.5 * findError.array().sum();

      if (iteration % 100 == 0 && m_trainVerbose == true)
      {
        std::cout << "At Iteration: " << iteration << " Error: " << error << std::endl;
      }

      // Shuffle inputs and target to same place
      shuffleSet(inputsWithBiasEntry, trainTargets, m_nData);
    }
  }

  void LayeredPerceptron::mlpfwd(const MatrixXd &inputs, int nData)
  {
    // arg inputs should be passed in with bias entry added
    // Go forward through hidden layer and output layer
    // Use logistic activation
    for (int n = 0; n < m_nHidden; n++)
    {
      m_hiddenLayer(nData, n) = 0;
      for (int m = 0; m < m_nIn + 1; m++)
      {
        auto connect = (inputs(nData, m) * m_weights1(m, n));
        m_hiddenLayer(nData, n) += connect;
      }
      // activation
      double resH = (1.0 / (1.0 + std::exp(-1.0 * m_beta * m_hiddenLayer(nData, n))));
      m_hiddenLayer(nData, n) = resH;
    }

    for (int o = 0; o < m_nOut; o++)
    {
      m_outputs(nData, o) = 0;
      for (int m = 0; m < m_nHidden + 1; m++)
      {
        auto connect = (m_hiddenLayer(nData, m) * m_weights2(m, o));
        m_outputs(nData, o) += connect;
      }
      // activation
      double resO = 1.0 / (1.0 + std::exp(-1.0 * m_beta * m_outputs(nData, o)));
      m_outputs(nData, o) = resO;
    }
  }

  double LayeredPerceptron::confmat(const MatrixXd &inputs, const MatrixXd &targets)
  {

    int nInputData = inputs.innerSize();
    ArrayXXd a(nInputData, m_nOut);
    ArrayXXd b(nInputData, m_nOut);
    MatrixXd biasInput(nInputData, 1);
    MatrixXd inputsWithBiasEntry(nInputData, m_nIn + 1);
    MatrixXd trainTargets = targets;
    a.fill(1.0);
    b.fill(0);
    biasInput.fill(-1.0);
    inputsWithBiasEntry.block(0, 0, nInputData, m_nIn) << inputs;
    inputsWithBiasEntry.col(m_nIn).tail(nInputData) << biasInput;

    m_hiddenLayer = MatrixXd(nInputData, m_nHidden + 1);
    m_hiddenLayer.block(0, 0, nInputData, m_nHidden).fill(0);
    m_outputs = MatrixXd(nInputData, m_nOut);
    m_outputs.fill(0);

    for (int nData = 0; nData < nInputData; nData++)
    {
      mlpfwd(inputsWithBiasEntry, nData);
    }

    int nClasses = trainTargets.outerSize();
    if (nClasses == 1)
    {
      nClasses = 2;
      m_outputs = (m_outputs.array() > 0.5).select(a, b);
    }
    else
    {
      // 1-of-N enoding
      D(std::cout << "network size: no. of classes " << nClasses << std::endl
                  << " nIn: " << m_nIn << ", nOut:" << m_nOut << ", nData: " << nInputData << std::endl;)
      m_outputs = indiceMax(m_outputs, nInputData, m_nOut);
      trainTargets = indiceMax(trainTargets, nInputData, m_nOut);

      D(std::cout << "Outputs As IndiceMax: " << std::endl
                  << (nInputData < 100 ? nInputData : 100) << ": " << m_outputs(seqN(0, nInputData < 100 ? nInputData : 100), all).transpose() << std::endl;)
      D(std::cout << "Targets As IndiceMax: " << std::endl
                  << (nInputData < 100 ? nInputData : 100) << ": " << trainTargets(seqN(0, nInputData < 100 ? nInputData : 100), all).transpose() << std::endl;)

      a = ArrayXXd(nInputData, 1);
      b = ArrayXXd(nInputData, 1);
      a.fill(1.0);
      b.fill(0);
    }

    MatrixXd cm(nClasses, nClasses);
    cm.fill(0);
    for (int i = 0; i < nClasses; i++)
    {
      for (int j = 0; j < nClasses; j++)
      {
        auto classSum = (((m_outputs.array() == i).select(a, b)) * ((trainTargets.array() == j).select(a, b))).sum();
        cm(i, j) = classSum;
      }
    }
    std::cout << "Confusion Matrix: " << std::endl
              << cm << std::endl;
    auto sumCM = cm.sum();
    double percCorrect;
    if (sumCM != 0)
    {
      percCorrect = cm.trace() / cm.sum();
      std::cout << cm.trace() / cm.sum() << std::endl;
    }
    else
    {
      percCorrect = cm.trace();
      std::cout << cm.trace() << std::endl;
    }
    return percCorrect;
  }

  ArrayXd LayeredPerceptron::indiceMax(const MatrixXd &matrix, int nData, int recordLength)
  {
    ArrayXd indices(nData);
    if (nData == 1)
    {
      ArrayXd Nrecord = matrix.reshaped();
      auto result = std::max_element(Nrecord.begin(), Nrecord.end());
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

  void shuffleSet(MatrixXd &inputs, MatrixXd &targets, int nData)
  {
    // Shuffle inputs and target to same place
    auto shuffle = [](int size)
    {
      std::vector<int> shuffled(size);
      std::iota(shuffled.begin(), shuffled.end(), 0);
      std::random_shuffle(shuffled.begin(), shuffled.end());
      return shuffled;
    };

    int countIndex = 0;
    MatrixXd tmpInputs = inputs;
    MatrixXd tmpTargets = targets;
    std::vector<int> shuffleIndex = shuffle(nData);
    for (int i = 0; i < shuffleIndex.size(); i++)
    {
      inputs.row(countIndex) = tmpInputs.row(shuffleIndex[i]);
      targets.row(countIndex) = tmpTargets.row(shuffleIndex[i]);
      countIndex++;
    }
  }

  void shuffleSet(MatrixXd &inputs, int nData)
  {
    // Shuffle inputs and target to same place
    auto shuffle = [](int size)
    {
      std::vector<int> shuffled(size);
      std::iota(shuffled.begin(), shuffled.end(), 0);
      std::random_shuffle(shuffled.begin(), shuffled.end());
      return shuffled;
    };

    int countIndex = 0;
    MatrixXd tmpInputs = inputs;
    std::vector<int> shuffleIndex = shuffle(nData);
    for (int i = 0; i < shuffleIndex.size(); i++)
    {
      inputs.row(countIndex) = tmpInputs.row(shuffleIndex[i]);
      countIndex++;
    }
  }

}
