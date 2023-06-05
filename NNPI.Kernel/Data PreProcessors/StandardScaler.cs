namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Standardizes the input data to have zero mean and unit variance.
    /// </summary>
    public class StandardScaler
    {
        private double[] _mean;
        private double[] _stdDev;

        /// <summary>
        /// Fits the StandardScaler to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of standardized data.</returns>
        public double[][] FitTransform(double[][] data)
        {
            int numRows = data.Length;
            int numCols = data[0].Length;

            _mean = new double[numCols];
            _stdDev = new double[numCols];

            double[][] standardizedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
                standardizedData[i] = new double[numCols];

            // Calculate the mean and standard deviation for each column
            for (int col = 0; col < numCols; col++)
            {
                double sum = 0;
                double squaredSum = 0;

                for (int row = 0; row < numRows; row++)
                {
                    sum += data[row][col];
                    squaredSum += Math.Pow(data[row][col], 2);
                }

                _mean[col] = sum / numRows;
                _stdDev[col] = Math.Sqrt((squaredSum / numRows) - Math.Pow(_mean[col], 2));
            }

            // Standardize the data in each column
            for (int row = 0; row < numRows; row++)
                for (int col = 0; col < numCols; col++)
                    standardizedData[row][col] = (data[row][col] - _mean[col]) / _stdDev[col];

            return standardizedData;
        }
    }
}
