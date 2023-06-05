namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Imputes missing values in the input data using the median value of the non-missing values in the same column.
    /// </summary>
    public class MedianImputer
    {
        private double[] _median;

        /// <summary>
        /// Fits the MedianImputer to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of imputed data.</returns>
        public double[][] FitTransform(double?[][] data)
        {
            int numRows = data.Length;
            int numCols = data[0].Length;

            _median = new double[numCols];

            double[][] imputedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                imputedData[i] = new double[numCols];
            }

            // Calculate the median for each column
            for (int col = 0; col < numCols; col++)
            {
                List<double> values = new List<double>();

                for (int row = 0; row < numRows; row++)
                {
                    if (data[row][col].HasValue)
                    {
                        values.Add(data[row][col].Value);
                    }
                }

                values.Sort();
                int count = values.Count;

                _median[col] = count % 2 == 0 ? (values[count / 2 - 1] + values[count / 2]) / 2 : values[count / 2];
            }

            // Impute missing values using the column median
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    imputedData[row][col] = data[row][col].HasValue ? data[row][col].Value : _median[col];
                }
            }

            return imputedData;
        }
    }
}
