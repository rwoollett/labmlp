#ifndef ML_LAYEREDPERCEPTRON_H
#define ML_LAYEREDPERCEPTRON_H

#include <Eigen/Dense>

using namespace Eigen;

#ifdef NDEBUG
#define D(x)
#else
#define D(x) x
#endif

namespace ML
{

  class LayeredPerceptron
  {
    int m_nIn;
    int m_nOut;
    int m_nData;
    int m_nHidden;
    double m_beta;
    double m_momentum;

    MatrixXd m_weights1;
    MatrixXd m_weights2;
    MatrixXd m_updatew1;
    MatrixXd m_updatew2;
    int m_threshold{0};

  public:
    // weight passed need i0 placement for bias input weights
    LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden);

    void mlptrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations);

    void mlpfwd(const MatrixXd &inputs, const MatrixXd &targets, double eta, int iteration);

    void mlpback(const MatrixXd &inputs, const MatrixXd &targets,
                 const MatrixXd &nOutputActivation, const MatrixXd &nHiddenActivationWithBias,
                 double eta, int nData)
    {
      //  sequential updates weight
      MatrixXd deltaO(1, m_nOut);
      MatrixXd deltaH(1, m_nHidden + 1);

      deltaO << (m_beta *
                 (nOutputActivation.row(nData) - targets.row(nData)).array() *
                 nOutputActivation.row(nData).array() *
                 (1.0 - nOutputActivation.row(nData).array()))
                    .eval();

      deltaH << ((nHiddenActivationWithBias.row(nData).array() * m_beta).array() *
                 (1.0 - nHiddenActivationWithBias.row(nData).array()).array() *
                 (deltaO * m_weights2.transpose()).array())
                    .eval();

      m_updatew1 = (eta * (inputs.row(nData).transpose() * deltaH(seqN(0, 1), seqN(0, m_nHidden))) + m_momentum * m_updatew1).eval();
      m_updatew2 = (eta * (nHiddenActivationWithBias.row(nData).transpose() * deltaO) + m_momentum * m_updatew2).eval();
      m_weights1 = (m_weights1 - m_updatew1).eval();
      m_weights2 = (m_weights2 - m_updatew2).eval();
    };

    void confmat(const MatrixXd &inputs, MatrixXd targets);

    ArrayXd indiceMax(const MatrixXd &matrix, int nData, int recordLength);
  };

}
#endif // ML_LAYEREDPERCEPTRON_H
