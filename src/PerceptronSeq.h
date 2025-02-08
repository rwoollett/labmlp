#ifndef ML_PERCEPTRONSEQ_H
#define ML_PERCEPTRONSEQ_H

#include <Eigen/Dense>

using namespace Eigen;

#ifdef NDEBUG
#define D(x)
#else
#define D(x) x
#endif

namespace ML::Seq
{

  class Perceptron
  {
    int m_nIn;
    int m_nOut;
    int m_nData;
    MatrixXd m_weights;
    int m_threshold{0};

  public:
    // weight passed need i0 placement for bias input weights
    Perceptron(const MatrixXd &inputs, const MatrixXd &targets);

    void pcntrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations);

    void pcnfwd(const MatrixXd &inputs, const MatrixXd &targets, double eta);

    void confmat(const MatrixXd &inputs, MatrixXd targets);

    ArrayXd indiceMax(const MatrixXd &matrix, int nData, int recordLength);
  };

}
#endif // ML_PERCEPTRONSEQ_H
