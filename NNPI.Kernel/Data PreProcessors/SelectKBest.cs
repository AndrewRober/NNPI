namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Selects the top K features with the highest scores according to a scoring function.
    /// </summary>
    public class SelectKBest
    {
        private readonly int _k;
        private readonly Func<double[][], double[][], double[]> _scoringFunction;

        /// <summary>
        /// Initializes a new instance of the SelectKBest class with the specified number of features to select and the scoring function.
        /// </summary>
        /// <param name="k">The number of top features to select.</param>
        /// <param name="scoringFunction">The scoring function to compute feature scores. It should take two 2D arrays (X and y) and return a 1D array of scores.</param>
        public SelectKBest(int k, Func<double[][], double[][], double[]> scoringFunction)
        {
            if (k <= 0)
            {
                throw new ArgumentException("The number of features to select (k) must be positive.", nameof(k));
            }

            _k = k;
            _scoringFunction = scoringFunction ?? throw new ArgumentNullException(nameof(scoringFunction));
        }

        /// <summary>
        /// Transforms the input data by selecting the top K features with the highest scores.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <param name="labels">A 2D array of target labels.</param>
        /// <returns>A 2D array of input data with only the top K features selected.</returns>
        public double[][] Transform(double[][] data, double[][] labels)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            if (labels == null || labels.Length == 0)
            {
                throw new ArgumentException("Labels must not be null or empty.", nameof(labels));
            }

            if (data.Length != labels.Length)
            {
                throw new ArgumentException("The number of data rows must equal the number of label rows.", nameof(labels));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;

            if (_k > numCols)
            {
                throw new ArgumentException("The number of features to select (k) must be less than or equal to the number of columns in the data.", nameof(_k));
            }

            // Compute feature scores
            double[] scores = _scoringFunction(data, labels);

            // Select the indices of the top K features
            var topKIndices = scores
                .Select((score, index) => (score, index))
                .OrderByDescending(x => x.score)
                .Take(_k)
                .Select(x => x.index)
                .ToList();

            double[][] transformedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                transformedData[i] = new double[_k];
            }

            // Copy the selected features to the transformed data
            for (int row = 0; row < numRows; row++)
            {
                for (int i = 0; i < _k; i++)
                {
                    transformedData[row][i] = data[row][topKIndices[i]];
                }
            }

            return transformedData;
        }
    }
}
