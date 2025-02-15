#include "LayeredPerceptron.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

namespace ML
{
  LayeredPerceptron::LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden)
      : m_nIn{1}, m_nOut{1}, m_nData{0}, m_nHidden{nhidden}, m_beta{1.0}, m_momentum{0.9}
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

    MatrixXd trainTargets = targets;

    D(std::cout << "train inputs: " << std::endl
                << inputs << std::endl;)
    D(std::cout << "train inputs with bias: " << std::endl
                << inputsWithBiasEntry << std::endl;)
    D(std::cout << "train targets " << std::endl
                << trainTargets << std::endl;)

    m_updatew1 = MatrixXd(m_nIn + 1, m_nHidden);
    m_updatew1.setZero();
    m_updatew2 = MatrixXd(m_nHidden + 1, m_nOut);
    m_updatew2.setZero();

    for (int i = 0; i < nIterations; i++)
    {
      mlpfwd(inputsWithBiasEntry, trainTargets, eta, i);

      // Shuffle inputs and target to same place
      auto shuffle = [](int size)
      {
        std::vector<int> shuffled(size);
        std::iota(shuffled.begin(), shuffled.end(), 0);
        std::random_shuffle(shuffled.begin(), shuffled.end());
        return shuffled;
      };

      int countIndex = 0;
      MatrixXd tmpInputs = inputsWithBiasEntry;
      MatrixXd tmpTargets = trainTargets;
      std::vector<int> shuffleIndex = shuffle(m_nData);
      for (int i = 0; i < shuffleIndex.size(); i++)
      {
        inputsWithBiasEntry.row(countIndex) = tmpInputs.row(shuffleIndex[i]);
        trainTargets.row(countIndex) = tmpTargets.row(shuffleIndex[i]);
        countIndex++;
      }
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

    // m_updatew1 = MatrixXd(m_nIn + 1, m_nHidden);
    // m_updatew1.setZero();
    // m_updatew2 = MatrixXd(m_nHidden + 1, m_nOut);
    // m_updatew2.setZero();

    for (int nData = 0; nData < m_nData; nData++)
    {
      for (int n = 0; n < m_nHidden; n++)
      {
        nHiddenActivationWithBias(nData, n) = 0;
        for (int m = 0; m < m_nIn + 1; m++)
        {
          auto connect = (inputs(nData, m) * m_weights1(m, n));
          nHiddenActivationWithBias(nData, n) += connect;
        }

        double resH = (1.0 / (1.0 + std::exp(-1.0 * m_beta * nHiddenActivationWithBias(nData, n))));
        nHiddenActivationWithBias(nData, n) = resH; 
      }

      // Now do output layer after hidden layer ; use logistic activation
      for (int o = 0; o < m_nOut; o++)
      {
        nOutputActivation(nData, o) = 0;
        for (int m = 0; m < m_nHidden + 1; m++)
        {
          auto connect = (nHiddenActivationWithBias(nData, m) * m_weights2(m, o));
          nOutputActivation(nData, o) += connect;
        }
        double resO = 1.0 / (1.0 + std::exp(-1.0 * m_beta * nOutputActivation(nData, o)));
        nOutputActivation(nData, o) = resO;
      }

      // std::cout << "nOutputActivation after fwd" << std::endl;

      mlpback(inputs, targets, nOutputActivation, nHiddenActivationWithBias, eta, nData);
    }

    MatrixXd findError(m_nData, m_nOut);
    findError << (nOutputActivation - targets).eval();
    findError = (findError.array().pow(2.0)).eval();
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

    // the following equationa are the fwd process of the mlp train
    // TODO: make the fwd process just do the fwd step and a new bckpropagate function in the train
    for (int nData = 0; nData < m_nData; nData++)
    {
      for (int n = 0; n < m_nHidden; n++)
      {
        double resH = (1.0 / (1.0 + std::exp(-1.0 * m_beta * nHiddenActivationWithBias(nData, n))));
        nHiddenActivationWithBias(nData, n) = resH; //(1.0 / (1.0 + std::exp(-1.0 * m_beta * nHiddenActivationWithBias(nData, n))));
      }
    }

    outputs = nHiddenActivationWithBias * m_weights2;

    for (int nData = 0; nData < m_nData; nData++)
    {
      for (int o = 0; o < m_nOut; o++)
      {
        double resO = 1.0 / (1.0 + std::exp(-1.0 * m_beta * outputs(nData, o)));
        outputs(nData, o) = resO;
      }
    }
    // end of fwd process - outputs are the activated results
    std::cout << "Outputs" << std::endl;
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
}
