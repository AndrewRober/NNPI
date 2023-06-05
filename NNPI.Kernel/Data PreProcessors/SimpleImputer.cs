namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Imputes missing values in the input data using a specified strategy.
    /// </summary>
    public class SimpleImputer
    {
        /// <summary>
        /// Imputation strategies for missing values.
        /// </summary>
        public enum ImputationStrategy
        {
            Mean,
            Median,
            MostFrequent
        }

        private readonly ImputationStrategy _strategy;

        /// <summary>
        /// Initializes a new instance of the SimpleImputer class with the specified imputation strategy.
        /// </summary>
        /// <param name="strategy">The imputation strategy to use. Default: ImputationStrategy.Mean.</param>
        public SimpleImputer(ImputationStrategy strategy = ImputationStrategy.Mean)
        {
            _strategy = strategy;
        }

        /// <summary>
        /// Imputes missing values in the input data using the specified strategy.
        /// </summary>
        /// <param name="data">A 2D array of input data with missing values.</param>
        /// <returns>A 2D array of input data with missing values imputed.</returns>
        public double[][] Transform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            double[][] imputedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                imputedData[i] = new double[numCols];
            }

            // Impute missing values in each column
            for (int col = 0; col < numCols; col++)
            {
                List<double> values = new List<double>();
                for (int row = 0; row < numRows; row++)
                {
                    if (!double.IsNaN(data[row][col]))
                    {
                        values.Add(data[row][col]);
                    }
                }

                double imputedValue;

                switch (_strategy)
                {
                    case ImputationStrategy.Median:
                        values.Sort();
                        imputedValue = values[values.Count / 2];
                        break;
                    case ImputationStrategy.MostFrequent:
                        imputedValue = values.GroupBy(v => v).OrderByDescending(g => g.Count()).First().Key;
                        break;
                    default: // ImputationStrategy.Mean
                        imputedValue = values.Average();
                        break;
                }

                for (int row = 0; row < numRows; row++)
                {
                    imputedData[row][col] = double.IsNaN(data[row][col]) ? imputedValue : data[row][col];
                }
            }

            return imputedData;
        }
    }
}
