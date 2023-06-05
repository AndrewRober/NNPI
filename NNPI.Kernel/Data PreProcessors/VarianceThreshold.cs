namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Removes features with variance below a specified threshold.
    /// </summary>
    public class VarianceThreshold
    {
        private double _threshold;

        /// <summary>
        /// Initializes a new instance of the VarianceThreshold class with the specified threshold.
        /// </summary>
        /// <param name="threshold">The variance threshold. Features with variance below this threshold will be removed. Default: 0.</param>
        public VarianceThreshold(double threshold = 0)
        {
            if (threshold < 0)
            {
                throw new ArgumentException("Threshold must be non-negative.", nameof(threshold));
            }

            _threshold = threshold;
        }

        /// <summary>
        /// Transforms the input data by removing features with variance below the specified threshold.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of input data with low-variance features removed.</returns>
        public double[][] Transform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            List<int> selectedColumns = new List<int>();

            // Select columns with variance above the threshold
            for (int col = 0; col < numCols; col++)
            {
                double mean = data.Select(row => row[col]).Average();
                double variance = data.Select(row => Math.Pow(row[col] - mean, 2)).Average();

                if (variance > _threshold)
                {
                    selectedColumns.Add(col);
                }
            }

            double[][] transformedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                transformedData[i] = new double[selectedColumns.Count];
            }

            // Copy the selected columns to the transformed data
            for (int row = 0; row < numRows; row++)
            {
                for (int i = 0; i < selectedColumns.Count; i++)
                {
                    transformedData[row][i] = data[row][selectedColumns[i]];
                }
            }

            return transformedData;
        }
    }
}
