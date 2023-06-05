namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Generates polynomial and interaction features for the input data.
    /// </summary>
    public class PolynomialFeatures
    {
        /// <summary>
        /// Gets the desired polynomial degree for the transformation.
        /// </summary>
        public int Degree { get; }

        /// <summary>
        /// Initializes a new instance of the PolynomialFeatures class.
        /// </summary>
        /// <param name="degree">The desired polynomial degree for the transformation.</param>
        public PolynomialFeatures(int degree)
        {
            if (degree < 0)
            {
                throw new ArgumentException("Degree must be non-negative.", nameof(degree));
            }

            Degree = degree;
        }

        /// <summary>
        /// Transforms the input data by generating polynomial and interaction features.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of transformed data with polynomial features.</returns>
        public double[][] Transform(double[][] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data must not be null or empty.", nameof(data));
            }

            int numRows = data.Length;
            int numCols = data[0].Length;
            int numFeatures = NumberOfOutputFeatures(numCols, Degree);

            double[][] transformedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                transformedData[i] = new double[numFeatures];
                int featureIdx = 0;
                for (int degree = 0; degree <= Degree; degree++)
                {
                    for (int n = 0; n <= degree; n++)
                    {
                        for (int k = 0; k < numCols; k++)
                        {
                            int exp1 = degree - n;
                            int exp2 = n;

                            // Compute the polynomial feature for the current column
                            transformedData[i][featureIdx] = Math.Pow(data[i][k], exp1) * (exp2 == 0 ? 1 : Math.Pow(data[i][(k + n) % numCols], exp2));
                            featureIdx++;
                        }
                    }
                }
            }

            return transformedData;
        }

        /// <summary>
        /// Calculates the number of output features for the given input features and polynomial degree.
        /// </summary>
        /// <param name="inputFeatures">The number of input features.</param>
        /// <param name="degree">The polynomial degree.</param>
        /// <returns>The number of output features.</returns>
        private int NumberOfOutputFeatures(int inputFeatures, int degree)
        {
            int outputFeatures = 0;
            for (int d = 0; d <= degree; d++)
                outputFeatures += (int)Math.Pow(inputFeatures, d) * (d + 1) / 2;
            return outputFeatures;
        }
    }


}
