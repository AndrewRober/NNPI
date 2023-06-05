namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// RandomUniformInitializer class that initializes the weights of a matrix with random values sampled from a uniform distribution.
    /// </summary>
    public class RandomUniformInitializer : WeightInitializer
    {
        private double minValue;
        private double maxValue;
        private Random random;

        /// <summary>
        /// Initializes a new instance of the RandomUniformInitializer class with the specified range for the uniform distribution.
        /// </summary>
        /// <param name="minValue">The lower bound of the uniform distribution. Default is -0.05.</param>
        /// <param name="maxValue">The upper bound of the uniform distribution. Default is 0.05.</param>
        public RandomUniformInitializer(double minValue = -0.05, double maxValue = 0.05)
        {
            this.minValue = minValue;
            this.maxValue = maxValue;
            random = new Random();
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values sampled from a uniform distribution with the specified range.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] = random.NextDouble() * (maxValue - minValue) + minValue;
                }
            }
        }
    }


}
