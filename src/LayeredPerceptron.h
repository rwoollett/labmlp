#ifndef ML_LAYEREDPERCEPTRON_H
#define ML_LAYEREDPERCEPTRON_H

#include <Eigen/Dense>
#include <iostream>

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

    MatrixXd m_hiddenLayer;
    MatrixXd m_outputs;
    MatrixXd m_weights1;
    MatrixXd m_weights2;
    MatrixXd m_updatew1;
    MatrixXd m_updatew2;
    int m_threshold{0};

  public:
    // weight passed need i0 placement for bias input weights
    LayeredPerceptron(const MatrixXd &inputs, const MatrixXd &targets, int nhidden);

    void mlptrain(const MatrixXd &inputs, const MatrixXd &targets, double eta, int nIterations);

    void earlystopping(const MatrixXd &inputs, const MatrixXd &targets,
                       const MatrixXd &valid, const MatrixXd &validtargets,
                       double eta, int nIterations)
    {
      int nValidData = valid.innerSize();
      MatrixXd biasInput(nValidData, 1);
      biasInput.fill(-1.0);

      // Add bias entry to valid
      MatrixXd validWithBiasEntry(nValidData, m_nIn + 1);
      validWithBiasEntry.block(0, 0, nValidData, m_nIn) << valid;
      validWithBiasEntry.col(m_nIn).tail(nValidData) << biasInput;

      double old_val_error1 = 100002.0;
      double old_val_error2 = 100001.0;
      double new_val_error = 100000.0;

      int count = 0;
      int totalIterations = 0;

      while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001))
      {
        m_hiddenLayer = MatrixXd(m_nData, m_nHidden + 1);
        m_hiddenLayer.block(0, 0, m_nData, m_nHidden).fill(0);
        m_outputs = MatrixXd(m_nData, m_nOut);
        m_outputs.fill(0);
 
        mlptrain(inputs, targets, eta, nIterations);
 
        old_val_error2 = old_val_error1;
        old_val_error1 = new_val_error;

        m_hiddenLayer = MatrixXd(nValidData, m_nHidden + 1);
        m_hiddenLayer.block(0, 0, nValidData, m_nHidden).fill(0);
        m_outputs = MatrixXd(nValidData, m_nOut);
        m_outputs.fill(0);
        for (int nData = 0; nData < nValidData; nData++)
        {
          mlpfwd(validWithBiasEntry, nData);
        }
        // new_val_error = 0.5*np.sum((validtargets-m_output)**2);
        MatrixXd findError(nValidData, m_nOut);
        findError << (validtargets - m_outputs).eval();
        findError = (findError.array().pow(2.0)).eval();
        new_val_error = 0.5 * findError.array().sum();
  
        totalIterations += nIterations;
      }

      std::cout << "Stopped" << std::endl << totalIterations << " " << new_val_error << " " << old_val_error1 << " " << old_val_error2 << std::endl;
      //return new_val_error

    }

    void mlpfwd(const MatrixXd &inputs, int nData);

    void mlpback(const MatrixXd &inputs, const MatrixXd &targets, int nData, double eta)
    {
      // arg inputs should be passed in with bias entry added
      //  sequential updates weight
      MatrixXd deltaO(1, m_nOut);
      MatrixXd deltaH(1, m_nHidden + 1);

      deltaO << (m_beta *
                 (m_outputs.row(nData) - targets.row(nData)).array() *
                 m_outputs.row(nData).array() *
                 (1.0 - m_outputs.row(nData).array()))
                    .eval();

      deltaH << ((m_hiddenLayer.row(nData).array() * m_beta).array() *
                 (1.0 - m_hiddenLayer.row(nData).array()).array() *
                 (deltaO * m_weights2.transpose()).array())
                    .eval();

      m_updatew1 = (eta * (inputs.row(nData).transpose() * deltaH(seqN(0, 1), seqN(0, m_nHidden))) + m_momentum * m_updatew1).eval();
      m_updatew2 = (eta * (m_hiddenLayer.row(nData).transpose() * deltaO) + m_momentum * m_updatew2).eval();
      m_weights1 = (m_weights1 - m_updatew1).eval();
      m_weights2 = (m_weights2 - m_updatew2).eval();
    };

    void confmat(const MatrixXd &inputs, const MatrixXd &targets);

    ArrayXd indiceMax(const MatrixXd &matrix, int nData, int recordLength);
  };

}
#endif // ML_LAYEREDPERCEPTRON_H
