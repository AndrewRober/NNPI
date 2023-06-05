using System.Numerics;

namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Scales the input data by transforming it to a specified range.
    /// </summary>
    public class MinMaxScaler
    {
        private double[] _min;
        private double[] _range;
        private readonly double _featureRangeMin;
        private readonly double _featureRangeMax;

        /// <summary>
        /// Initializes a new instance of the MinMaxScaler class.
        /// </summary>
        /// <param name="featureRangeMin">The lower bound of the desired feature range.</param>
        /// <param name="featureRangeMax">The upper bound of the desired feature range.</param>
        public MinMaxScaler(double featureRangeMin = 0, double featureRangeMax = 1)
        {
            if (featureRangeMin >= featureRangeMax)
            {
                throw new ArgumentException("Feature range minimum must be less than maximum.");
            }

            _featureRangeMin = featureRangeMin;
            _featureRangeMax = featureRangeMax;
        }

        /// <summary>
        /// Fits the scaler to the input data and transforms the data using the calculated scaling factors.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of transformed data with min-max scaling applied.</returns>
        public double[][] FitTransform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            _min = new double[numCols];
            _range = new double[numCols];

            double[][] scaledData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                scaledData[i] = new double[numCols];
            }

            // Calculate the minimum and range for each column
            for (int col = 0; col < numCols; col++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;

                for (int row = 0; row < numRows; row++)
                {
                    double value = data[row][col];
                    min = Math.Min(min, value);
                    max = Math.Max(max, value);
                }

                _min[col] = min;
                _range[col] = max - min;
            }

            // Scale the data using the minimum and range
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    scaledData[row][col] = (_range[col] == 0) ? _featureRangeMin : _featureRangeMin + (data[row][col] - _min[col]) / _range[col] * (_featureRangeMax - _featureRangeMin);
                }
            }

            return scaledData;
        }
    }


/// <summary>
/// Reduces the dimensionality of input data by projecting it onto the principal components.
/// </summary>
public class PCA
    {
        private readonly int _nComponents;
        private Matrix<double> _projectionMatrix;

        /// <summary>
        /// Initializes a new instance of the PCA class with the specified number of components.
        /// </summary>
        /// <param name="nComponents">The number of components to keep. Must be positive.</param>
        public PCA(int nComponents)
        {
            if (nComponents <= 0)
            {
                throw new ArgumentException("The number of components must be positive.", nameof(nComponents));
            }

            _nComponents = nComponents;
        }

        /// <summary>
        /// Fits the PCA model to the input data.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        public void Fit(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            if (_nComponents > numCols)
            {
                throw new ArgumentException("The number of components must be less than or equal to the number of columns in the data.", nameof(_nComponents));
            }

            Matrix<double> dataMatrix = DenseMatrix.OfArray(data);

            // Center the data
            Vector<double> columnMeans = dataMatrix.ColumnSums() / numRows;
            dataMatrix = (dataMatrix - columnMeans.ToRowMatrix()).Divide(Math.Sqrt(numRows - 1));

            // Compute the covariance matrix
            Matrix<double> covarianceMatrix = dataMatrix.Transpose() * dataMatrix;

            // Compute the eigenvalues and eigenvectors of the covariance matrix
            var eigen = covarianceMatrix.Evd();

            // Sort the eigenvectors by descending eigenvalues
            var sortedIndices = eigen.EigenValues.Select((val, idx) => (val, idx)).OrderByDescending(x => x.val.Magnitude).Select(x => x.idx).ToArray();
            Matrix<double> sortedEigenVectors = DenseMatrix.Create(numCols, numCols, (i, j) => eigen.EigenVectors[i, sortedIndices[j]]);

            // Select the top n eigenvectors
            _projectionMatrix = sortedEigenVectors.SubMatrix(0, numCols, 0, _nComponents);
        }

        /// <summary>
        /// Transforms the input data by projecting it onto the principal components.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of transformed input data.</returns>
        public double[][] Transform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            if (_projectionMatrix == null)
            {
                throw new InvalidOperationException("The PCA model must be fitted before transforming data.");
            }

            Matrix<double> dataMatrix = DenseMatrix.OfArray(data);
            Matrix<double> transformedDataMatrix = dataMatrix * _projectionMatrix;

            return transformedDataMatrix.ToRowArrays();
        }


    }
}
