namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Imputes missing values in the input data using the mean value of the non-missing values in the same column.
    /// </summary>
    public class MeanImputer
    {
        private double[] _mean;

        /// <summary>
        /// Fits the MeanImputer to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of imputed data.</returns>
        public double[][] FitTransform(double?[][] data)
        {
            int numRows = data.Length;
            int numCols = data[0].Length;

            _mean = new double[numCols];

            double[][] imputedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                imputedData[i] = new double[numCols];
            }

            // Calculate the mean for each column
            for (int col = 0; col < numCols; col++)
            {
                double sum = 0;
                int count = 0;

                for (int row = 0; row < numRows; row++)
                {
                    if (data[row][col].HasValue)
                    {
                        sum += data[row][col].Value;
                        count++;
                    }
                }

                _mean[col] = sum / count;
            }

            // Impute missing values using the column mean
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    imputedData[row][col] = data[row][col].HasValue ? data[row][col].Value : _mean[col];
                }
            }

            return imputedData;
        }
    }
}
