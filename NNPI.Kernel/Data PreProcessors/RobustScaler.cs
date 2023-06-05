namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Scales the input data using the interquartile range and median for robustness against outliers.
    /// </summary>
    public class RobustScaler
    {
        private double[] _median;
        private double[] _iqr;

        /// <summary>
        /// Fits the scaler to the input data and transforms the data using the calculated scaling factors.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of transformed data with robust scaling applied.</returns>
        public double[][] FitTransform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            _median = new double[numCols];
            _iqr = new double[numCols];

            double[][] scaledData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                scaledData[i] = new double[numCols];
            }

            // Calculate the median and interquartile range for each column
            for (int col = 0; col < numCols; col++)
            {
                List<double> values = new List<double>();
                for (int row = 0; row < numRows; row++)
                {
                    values.Add(data[row][col]);
                }

                values.Sort();

                double median = values[numRows / 2];
                double q1 = values[numRows / 4];
                double q3 = values[(3 * numRows) / 4];
                double iqr = q3 - q1;

                _median[col] = median;
                _iqr[col] = iqr;
            }

            // Scale the data using the median and interquartile range
            for (int row = 0; row < numRows; row++)
                for (int col = 0; col < numCols; col++)
                    scaledData[row][col] = (_iqr[col] == 0) ? 0 : (data[row][col] - _median[col]) / _iqr[col];

            return scaledData;
        }
    }
}
