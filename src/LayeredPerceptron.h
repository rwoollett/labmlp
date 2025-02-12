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

    MatrixXd m_weights1;
    MatrixXd m_weights2;
    int m_threshold{0};

  public:
    // weight passed need i0 placement for bias input weights
    LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden, double beta);

    void mlptrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations);

    void mlpfwd(const MatrixXd &inputs, const MatrixXd &targets, double eta, int iteration);

    void mlpbkwdpropogate(const MatrixXd &inputs, const MatrixXd &targets, double eta, int iteration);

    void confmat(const MatrixXd &inputs, MatrixXd targets);

    ArrayXd indiceMax(const MatrixXd &matrix, int nData, int recordLength);
  };

}
#endif // ML_LAYEREDPERCEPTRON_H
