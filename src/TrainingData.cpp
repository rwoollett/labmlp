#include "TrainingData.h"
#include "io_utility/io_utility.h"
#include <array>

using namespace io_utility;

namespace ML::DataSet
{

  void trainIrisMLP()
  {
    // The iris datafile
    //
    MatrixXd dataSet;
    std::vector<int> takeCols{0, 1, 2, 3, 4}; // 0-index of cols to read from file.
    try
    {
      dataSet = readDataFile("../dataset/iris_proc.data", takeCols);
    }
    catch (std::string e)
    {
      std::cout << "trainPima error: " << e << std::endl;
      return;
    }

    std::cout << dataSet.innerSize() << " " << dataSet.outerSize() << std::endl;
    int amountN = dataSet.innerSize();
    int nDataSetCols = dataSet.cols();

    // Preprocessing steps
    // Do standardise on data
    MatrixXd trainToStandardize = dataSet;
    // define standardize col - all except the classes in last cell
    MatrixXd result;
    for (int i = 0; i < nDataSetCols - 1; i++)
    {
      MatrixXd result = standardizeColumn(trainToStandardize.col(i), amountN, NormalizationType::MAXIMUM);
      trainToStandardize.col(i) = result;
    }

    D(std::cout << "training set before normal standardization" << std::endl;)
    D(std::cout << dataSet(seqN(0, 5), all) << std::endl;)
    D(std::cout << "training set with mornal standardization" << std::endl;)
    D(std::cout << trainToStandardize(seqN(0, 5), all) << std::endl;)

    dataSet = trainToStandardize;

    // Add encoded targets to dataset
    MatrixXd targetsToEncode = dataSet.col(nDataSetCols - 1);
    MatrixXd targetEncoded(amountN, 3);
    targetEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      int irisClass = static_cast<int>(targetsToEncode(i, 0));
      if (irisClass == 0)
      {
        targetEncoded(i, 0) = 1;
      }
      else if (irisClass == 1)
      {
        targetEncoded(i, 1) = 1;
      }
      else if (irisClass == 2)
      {
        targetEncoded(i, 2) = 1;
      }
    }

    shuffleSet(dataSet, targetEncoded, amountN);


    MatrixXd trainTargets = targetEncoded(seqN(1, amountN / 2, 2), all);
    MatrixXd tmpTargets = targetEncoded(seqN(0, amountN / 2, 2), all);

    MatrixXd trainInputs = dataSet(seqN(1, amountN / 2, 2), seqN(0, dataSet.outerSize() - 1));
    MatrixXd tmpInputs = dataSet(seqN(0, amountN / 2, 2), seqN(0, dataSet.outerSize() - 1));

    int tmpAmountN = amountN / 2;
    MatrixXd testInputs = tmpInputs(seqN(1, tmpAmountN / 2, 2), seqN(0, tmpInputs.outerSize()));
    MatrixXd validInputs = tmpInputs(seqN(0, tmpAmountN / 2, 2), seqN(0, tmpInputs.outerSize()));
    MatrixXd testTargets = tmpTargets(seqN(1, tmpAmountN / 2, 2), all);
    MatrixXd validTargets = tmpTargets(seqN(0, tmpAmountN / 2, 2), all);

    std::cout << "Training set: " << trainInputs.innerSize() << " Training target: " << trainTargets.innerSize() << std::endl;
    std::cout << "Testing set: " << testInputs.innerSize() << "  Testing target: " << testTargets.innerSize() << std::endl;
    std::cout << "Valid set: " << validInputs.innerSize() << " Valid target: " << validTargets.innerSize() << std::endl;
    std::cout << "Training set: " << trainInputs.outerSize() << " Training target: " << trainTargets.outerSize() << std::endl;
    std::cout << "Testing set: " << testInputs.outerSize() << "  Testing target: " << testTargets.outerSize() << std::endl;
    std::cout << "Valid set: " << validInputs.outerSize() << " Valid target: " << validTargets.outerSize() << std::endl;

    int nHidden = 2;
    int nIterations = 101;
    double learningRateETA = 0.1;

    LayeredPerceptron mlp(trainInputs, trainTargets, nHidden);
    mlp.mlptrain(trainInputs, trainTargets, learningRateETA, nIterations);
    mlp.earlystopping(trainInputs, trainTargets, validInputs, validTargets, learningRateETA, nIterations);

    mlp.confmat(testInputs, testTargets);
  }

  void trainPimaSeq()
  {
    // The pima datafile into the data for pcn
    //           -  3 is: 4. Triceps skin fold thickness (mm)                * not much improvement with this
    //           -  5 is: 6. Body mass index (weight in kg/(height in m)^2)  * not much improvement with this
    //
    MatrixXd dataSet;
    // dataset col indice are:
    //  0)         0 is: 1. Number of times pregnant                  [CAPPED at 8]
    //  1)         1 is: 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test [ENCODED 1ofN]
    //  2)         2 is: 3. Diastolic blood pressure (mm Hg)          [ENCODED 1ofN]
    //  3)         4 is: 5. 2-Hour serum insulin (mu U/ml)            [ENCODED 1ofN]
    //  4)         6 is: 7. Diabetes pedigree function                [ENCODED 1ofN]
    //  5)         7 is: 8. Age (years)                               [ENCODED 1ofN]
    //  6)  target 8 is: 9. Class variable (0 or 1)
    std::vector<int> takeCols{0, 1, 2, 4, 6, 7, 8}; // 0-index of cols to read from file.

    int capPregnant = 0;
    int encodeGlucose = 1;
    int encodeBloodPressureCol = 2;
    int encode2HrSerum = 3;
    int encodePedigree = 4;
    int encodeAgeCol = 5;

    int noleftColsToClasses = 6; // Before the encoded cols are merged together in final dataset

    try
    {
      dataSet = readDataFile("../dataset/pima-indians-diabetes.data", takeCols);
    }
    catch (std::string e)
    {
      std::cout << "trainPima error: " << e << std::endl;
      return;
    }

    std::cout << dataSet.innerSize() << " " << dataSet.outerSize() << std::endl;
    int amountN = dataSet.innerSize();
    double learningRateETA = 0.25;
    int noIterations = 100;

    // Preprocessing steps

    // Some records in data set have 0 (all missed entries).
    // These records can be removed.
    // dataSet = cleanSparseRecords(dataSet, amountN, noleftColsToClasses);
    std::cout << dataSet.innerSize() << " " << dataSet.outerSize() << std::endl;
    amountN = dataSet.innerSize();

    // Data such Triceps skin fold thickness (mm) 1ofN encoded into six categories
    // Now 3. Diastolic blood pressure (mm Hg)
    MatrixXd encodeBloodPressure = dataSet.col(encodeBloodPressureCol);
    MatrixXd bloodPressureEncoded(amountN, 5);
    bloodPressureEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      int bloodHg = static_cast<int>(encodeBloodPressure(i, 0));
      if (bloodHg <= 50)
      {
        bloodPressureEncoded(i, 0) = 1;
      }
      else if (bloodHg > 50 && bloodHg <= 70)
      {
        bloodPressureEncoded(i, 1) = 1;
      }
      else if (bloodHg > 70 && bloodHg <= 90)
      {
        bloodPressureEncoded(i, 2) = 1;
      }
      else if (bloodHg > 90 && bloodHg <= 110)
      {
        bloodPressureEncoded(i, 3) = 1;
      }
      else if (bloodHg > 110)
      {
        bloodPressureEncoded(i, 4) = 1;
      }
    }

    // Data such as age can be 1ofN encoded into five categories
    MatrixXd encodeAge = dataSet.col(encodeAgeCol);
    MatrixXd ageEncoded(amountN, 5);
    ageEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      int age = static_cast<int>(encodeAge(i, 0));
      if (age <= 30)
      {
        ageEncoded(i, 0) = 1;
      }
      else if (age > 30 && age <= 40)
      {
        ageEncoded(i, 1) = 1;
      }
      else if (age > 40 && age <= 50)
      {
        ageEncoded(i, 2) = 1;
      }
      else if (age > 50 && age <= 60)
      {
        ageEncoded(i, 3) = 1;
      }
      else if (age > 60)
      {
        ageEncoded(i, 4) = 1;
      }
    }

    // Look at: 2-Hour serum insulin (mu U/ml) - encode it too
    // 6 categories
    MatrixXd encodeSerum = dataSet.col(encode2HrSerum);
    MatrixXd serumEncoded(amountN, 6);
    serumEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      int serum = static_cast<int>(encodeSerum(i, 0));
      if (serum <= 100)
      {
        serumEncoded(i, 0) = 1;
      }
      else if (serum > 100 && serum <= 200)
      {
        serumEncoded(i, 1) = 1;
      }
      else if (serum > 200 && serum <= 300)
      {
        serumEncoded(i, 2) = 1;
      }
      else if (serum > 300 && serum <= 400)
      {
        serumEncoded(i, 3) = 1;
      }
      else if (serum > 400 && serum <= 500)
      {
        serumEncoded(i, 4) = 1;
      }
      else if (serum > 500)
      {
        serumEncoded(i, 5) = 1;
      }
    }

    // Look at: Glucose - encode it too
    // 4 categories
    MatrixXd encodeMatGlucose = dataSet.col(encodeGlucose);
    MatrixXd glucoseEncoded(amountN, 4);
    glucoseEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      int glucose = static_cast<int>(encodeMatGlucose(i, 0));
      if (glucose <= 75)
      {
        glucoseEncoded(i, 0) = 1;
      }
      else if (glucose > 75 && glucose <= 125)
      {
        glucoseEncoded(i, 1) = 1;
      }
      else if (glucose > 125 && glucose <= 175)
      {
        glucoseEncoded(i, 2) = 1;
      }
      else if (glucose > 175)
      {
        glucoseEncoded(i, 3) = 1;
      }
    }

    // Look at: Diabetes pedigree function
    // 6 categories
    MatrixXd encodeMatPedigree = dataSet.col(encodePedigree);
    MatrixXd pedigreeEncoded(amountN, 6);
    pedigreeEncoded.setZero();
    for (int i = 0; i < amountN; i++)
    {
      double pedigree = encodeMatPedigree(i, 0);
      if (pedigree <= 0.25)
      {
        pedigreeEncoded(i, 0) = 1;
      }
      else if (pedigree > 0.25 && pedigree <= 0.5)
      {
        pedigreeEncoded(i, 1) = 1;
      }
      else if (pedigree > 0.5 && pedigree <= 0.75)
      {
        pedigreeEncoded(i, 2) = 1;
      }
      else if (pedigree > 0.75 && pedigree <= 1.0)
      {
        pedigreeEncoded(i, 3) = 1;
      }
      else if (pedigree > 1.0 && pedigree <= 1.25)
      {
        pedigreeEncoded(i, 4) = 1;
      }
      else if (pedigree > 1.25)
      {
        pedigreeEncoded(i, 5) = 1;
      }
    }

    // Look at: Number of times pregnant
    // Cap pregnant amount above 8 at 8.
    MatrixXd pregnantCapped = dataSet.col(capPregnant);
    for (int i = 0; i < amountN; i++)
    {
      int pregnantAmt = pregnantCapped(i, 0);

      if (pregnantAmt >= 8)
      {
        pregnantCapped(i, 0) = 8;
      }
    }

    // Any encoded columns with extra cells are now merge to the new dataset
    // 1) Skin fold (mm)
    // 2) Plasma glucose concentration
    // 3) 2Hr Serum (mu U/ml)
    // 4) Diabetes pedigree
    // 5) Age

    // Create a new matrix with the appropriate size   // Amount of encoded cells
    D(std::cout << dataSet.cols() << std::endl;)
    int newDataSetCols = dataSet.cols() +
                         (glucoseEncoded.cols() - 1) +       // 4
                         (bloodPressureEncoded.cols() - 1) + // 5
                         (serumEncoded.cols() - 1) +         // 6
                         (pedigreeEncoded.cols() - 1) +      // 6
                         (ageEncoded.cols() - 1);            // 5

    MatrixXd newDataSet(amountN, newDataSetCols);
    newDataSet << pregnantCapped,
        glucoseEncoded,
        bloodPressureEncoded,
        serumEncoded,
        pedigreeEncoded,
        ageEncoded,
        dataSet.col(6);

    // Do standardise on data
    MatrixXd trainToStandardize = newDataSet;
    // define standardize col - all except the classes in last cell
    MatrixXd result;
    for (int i = 0; i < newDataSetCols - 1; i++)
    {
      MatrixXd result = standardizeColumn(trainToStandardize.col(i), amountN, NormalizationType::VARIANCE);
      trainToStandardize.col(i) = result;
    }

    std::cout << "trainToStandardize done" << std::endl;
    D(std::cout << trainToStandardize << std::endl;)
    dataSet = trainToStandardize;

    MatrixXd trainTargets = dataSet(seqN(1, amountN / 2, 2), last);
    MatrixXd tmpTargets = dataSet(seqN(0, amountN / 2, 2), last);

    MatrixXd trainInputs = dataSet(seqN(1, amountN / 2, 2), seqN(0, dataSet.outerSize() - 1));
    MatrixXd tmpInputs = dataSet(seqN(0, amountN / 2, 2), seqN(0, dataSet.outerSize() - 1));

    int tmpAmountN = amountN / 2;
    MatrixXd testInputs = tmpInputs(seqN(1, tmpAmountN / 2, 2), seqN(0, tmpInputs.outerSize()));
    MatrixXd validInputs = tmpInputs(seqN(0, tmpAmountN / 2, 2), seqN(0, tmpInputs.outerSize()));
    MatrixXd testTargets = tmpTargets(seqN(1, tmpAmountN / 2, 2), last);
    MatrixXd validTargets = tmpTargets(seqN(0, tmpAmountN / 2, 2), last);

    // Seq::Perceptron pcn(trainInputs, trainTargets);
    // pcn.pcntrain(trainInputs, trainTargets, learningRateETA, noIterations);
    // pcn.confmat(testInputs, testTargets);

    std::cout << "Training set: " << trainInputs.innerSize() << " Training target: " << trainTargets.innerSize() << std::endl;
    std::cout << "Testing set: " << testInputs.innerSize() << "  Testing target: " << testTargets.innerSize() << std::endl;
    std::cout << "Valid set: " << validInputs.innerSize() << " Valid target: " << validTargets.innerSize() << std::endl;
    std::cout << "Training set: " << trainInputs.outerSize() << " Training target: " << trainTargets.outerSize() << std::endl;
    std::cout << "Testing set: " << testInputs.outerSize() << "  Testing target: " << testTargets.outerSize() << std::endl;
    std::cout << "Valid set: " << validInputs.outerSize() << " Valid target: " << validTargets.outerSize() << std::endl;

    int nHidden = 2;
    int nIterations = 51;

    LayeredPerceptron mlp(trainInputs, trainTargets, nHidden);
    mlp.mlptrain(trainInputs, trainTargets, learningRateETA, nIterations);
    mlp.earlystopping(trainInputs, trainTargets, validInputs, validTargets, learningRateETA, nIterations);

    mlp.confmat(testInputs, testTargets);
  }

  void testTrainNClasses()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(20, 10);
    trainInputs.fill(0.0);
    MatrixXd trainTargets(20, 10);
    trainTargets.fill(0.0);

    for (int i = 0; i < 10; i++)
    {
      trainInputs(i, i) = 1.0;
    }
    for (int i = 0; i < 10; i++)
    {
      trainInputs(i + 10, i) = 1.0;
    }
    for (int i = 0; i < 10; i++)
    {
      trainTargets(i, i) = 1.0;
    }
    for (int i = 0; i < 10; i++)
    {
      trainTargets(i + 10, i) = 1.0;
    }

    std::cout << "Train inputs" << std::endl;
    std::cout << trainInputs << std::endl;
    std::cout << "Train targets" << std::endl;
    std::cout << trainTargets << std::endl;

    // Seq::Perceptron pcn(trainInputs, trainTargets);
    // pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

    // pcn.confmat(trainInputs, trainTargets);
    // Seq::Perceptron pcn(trainInputs, trainTargets);
    // pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

    // pcn.confmat(trainInputs, trainTargets);

    int nHidden = 10;
    int nIterations = 101;

    LayeredPerceptron mlp(trainInputs, trainTargets, nHidden);
    mlp.mlptrain(trainInputs, trainTargets, learningRateETA, nIterations);

    mlp.confmat(trainInputs, trainTargets);
  }

  void trainOr()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(4, 2);
    trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 1.0;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 6);

    pcn.confmat(trainInputs, trainTargets);
  }

  void trainXOr()
  {
    double learningRateETA = 0.25;

    MatrixXd trainInputs(4, 3);
    trainInputs << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;
    // MatrixXd trainInputs(4, 2);
    // trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;

    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 0.0;

    Perceptron pcn(trainInputs, trainTargets);
    pcn.pcntrain(trainInputs, trainTargets, learningRateETA, 14);

    pcn.confmat(trainInputs, trainTargets);
  }

  void trainXOrMLP()
  {
    double learningRateETA = 0.25;
    int nHidden = 2;
    int nIterations = 1001;

    MatrixXd trainInputs(4, 2);
    trainInputs << 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;
    MatrixXd trainTargets(4, 1);
    trainTargets << 0.0, 1.0, 1.0, 0.0;

    LayeredPerceptron mlp(trainInputs, trainTargets, nHidden);
    // mlp.mlptrain(trainInputs, trainTargets, learningRateETA, nIterations);
    // mlp.confmat(trainInputs, trainTargets);

    mlp.earlystopping(trainInputs, trainTargets, trainInputs, trainTargets, learningRateETA, 100);
    mlp.confmat(trainInputs, trainTargets);
  }

  MatrixXd readDataFile(std::string fileName, const std::vector<int> &takeCols)
  {
    MatrixXd dataSet(100, 10);
    dataSet.fill(1.1);
    int countSet = 0;
    int inputCount = 0;
    int takeIndex = 0;
    char ch = ',';

    std::vector<std::vector<double>> dataList;

    // pima = np.loadtxt('pima-indians-diabetes.data',delimiter=',')
    // np.shape(pima)
    // int dataSize;
    // int recordLength;

    auto [dataSize, recordLength] = readDataShapeFromFile(fileName);
    if (dataSize == 0)
    {
      throw std::string("No data found at ") + fileName;
    }
    std::cout << "Data file size: " << dataSize << " " << recordLength << std::endl;

    // using takeCols make the read dataSet use size of takeCols
    dataSet = MatrixXd(dataSize, takeCols.size());

    auto fileLines = read_file(fileName, true);
    for (std::string fileLine : fileLines)
    {
      auto stripLine = stripFileLine(fileLine);
      if (isCommentLine(stripLine))
      {
        std::cout << stripLine << std::endl;
        continue;
      }
      auto dataLineArray = splitStringToDouble(stripLine, ch);
      inputCount = dataLineArray.size();
      takeIndex = 0;

      if (inputCount > 0)
      {
        ArrayXd rec(takeCols.size());

        for (int i = 0; i < inputCount; i++)
        {
          if (auto it = std::find(takeCols.begin(), takeCols.end(), i) != takeCols.end())
          {
            dataSet(countSet, takeIndex) = dataLineArray[i];
            takeIndex++;
          }
        }
      }
      countSet++;
    }

    return dataSet;
  }

  std::string stripFileLine(std::string line)
  {
    char ch = '\r';
    std::string stripLine("");
    auto it = std::find(line.begin(), line.end(), ch);
    stripLine.resize(std::distance(line.begin(), it));
    std::copy(line.begin(), it, stripLine.begin());
    return stripLine;
  }

  bool isCommentLine(std::string line)
  {
    char ch = '#';
    bool isCommentLine = false;
    auto it = std::find(line.begin(), line.end(), ch);
    if (it != line.end())
    {
      isCommentLine = true;
    }
    return isCommentLine;
  }

  std::vector<double> splitStringToDouble(const std::string &str, char delimiter)
  {
    std::vector<double> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter))
    {
      tokens.push_back(std::stod(token));
    }

    return tokens;
  }

  std::tuple<int, int> readDataShapeFromFile(std::string fileName)
  {
    int countSet = 0;
    int inputCount = 0;
    char ch = ',';
    auto fileLines = read_file(fileName, true);

    for (std::string fileLine : fileLines)
    {
      auto stripLine = stripFileLine(fileLine);
      if (isCommentLine(stripLine))
      {
        std::cout << stripLine << std::endl;
        continue;
      }
      auto dataLineArray = splitStringToDouble(stripLine, ch);
      if (inputCount > 0)
      {
        if (dataLineArray.size() != inputCount)
        {
          std::cout << "Found discripancy in data input line sizes!" << std::endl;
        }
      }
      inputCount = dataLineArray.size();
      countSet++;
    }

    return {countSet, inputCount};
  }

  MatrixXd standardizeColumn(const MatrixXd &col, int nData, NormalizationType normType)
  {
    MatrixXd workCol(nData, 1);
    MatrixXd meanV(nData, 1);
    MatrixXd tosqrtd(nData, 1);
    MatrixXd sqrtd(nData, 1);
    MatrixXd normalize(nData, 1);

    workCol << col;

    D(std::cout << "workcol" << std::endl;)
    D(std::cout << workCol(seqN(0, 5), all) << std::endl;)
    // Col minus mean of nData columns
    double colMean = workCol.mean();
    double sumCol = workCol.sum();
    meanV.fill(colMean);

    // Update the workCol with col - mean.
    // Later the workCol is divided the variance.
    // col = col - mean
    workCol -= meanV;
    D(std::cout << "workcol - mean" << std::endl;)
    D(std::cout << workCol(seqN(0, 5), all) << std::endl;)

    // Col() now had mean minus :
    // Can normalize either by variance or the maximum
    switch (normType)
    {
    case NormalizationType::VARIANCE:
    {
      tosqrtd << workCol;
      sqrtd = tosqrtd.array().square();
      double sumSquares = sqrtd.sum();
      double variance = sumSquares / (nData - 1);
      D(std::cout << "sumSquares:         " << sumSquares << std::endl;)
      D(std::cout << "variance of col(0): " << variance << std::endl;)
      // Workcol has minus mean now divid variance calculated.
      // col = (col - mean)/variance
      normalize = workCol / variance;

      D(std::cout << "normalize with variance" << std::endl;)
      D(std::cout << normalize(seqN(0, 5), all) << std::endl;)

      break;
    }
    case NormalizationType::MAXIMUM:
    {
      auto max = workCol.maxCoeff();
      auto min = workCol.minCoeff();
      auto imax = std::max(max, std::abs(min));

      D(std::cout << "max:" << max << " min" << min << std::endl;)
      // 2) div workCols / imax to normalize
      normalize = workCol / imax;
      D(std::cout << "normalize with max" << std::endl;)
      D(std::cout << normalize(seqN(0, 5), all) << std::endl;)
      break;
    }
    default:
      // shound get here;
      break;
    }

    return normalize;
  }

  MatrixXd cleanSparseRecords(const MatrixXd &dataSet, int amountN, int leftCols)
  {
    MatrixXd filteredDataSet(amountN, dataSet.cols());
    int filteredRowCount = 0;

    for (int i = 0; i < amountN; ++i)
    {
      bool hasZero = false;
      for (int j = 0; j < leftCols; ++j)
      {
        if (dataSet(i, j) == 0)
        {
          hasZero = true;
          break;
        }
      }
      if (!hasZero)
      {
        filteredDataSet.row(filteredRowCount++) = dataSet.row(i);
      }
    }
    // Resize the filteredDataSet to the actual number of rows
    filteredDataSet.conservativeResize(filteredRowCount, Eigen::NoChange);

    // std::cout << "Filtered DataSet:" << std::endl;
    return filteredDataSet;
  }
}