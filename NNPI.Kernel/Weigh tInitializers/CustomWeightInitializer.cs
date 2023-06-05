namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// CustomWeightInitializer class that initializes the weights of a matrix with random values within a specified range.
    /// </summary>
    public class CustomWeightInitializer : WeightInitializer
    {
        private Random random;
        private double lowerBound;
        private double upperBound;

        /// <summary>
        /// Initializes a new instance of the CustomWeightInitializer class.
        /// </summary>
        /// <param name="lowerBound">The lower bound of the random values.</param>
        /// <param name="upperBound">The upper bound of the random values.</param>
        public CustomWeightInitializer(double lowerBound, double upperBound)
        {
            random = new Random();
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values within the specified range.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weights[i, j] = random.NextDouble() * (upperBound - lowerBound) + lowerBound;
                }
            }
        }
    }
}
