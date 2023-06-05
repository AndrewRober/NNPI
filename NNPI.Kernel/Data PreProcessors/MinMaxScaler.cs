using System.Numerics;

using NNPI.Kernel.NNPIMath.Linear_Algebra;

using Vector = NNPI.Kernel.NNPIMath.Linear_Algebra.Vector;

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
}
