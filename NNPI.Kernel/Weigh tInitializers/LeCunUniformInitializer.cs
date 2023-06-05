namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// LeCunUniformInitializer class that initializes the weights of a matrix with random values sampled from a uniform distribution
    /// with a range defined by the LeCun initialization method.
    /// </summary>
    public class LeCunUniformInitializer : WeightInitializer
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the LeCunUniformInitializer class.
        /// </summary>
        public LeCunUniformInitializer()
        {
            random = new Random();
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values sampled from a uniform distribution
        /// with a range defined by the LeCun initialization method.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);

            double limit = Math.Sqrt(3.0 / inputSize);

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weights[i, j] = random.NextDouble() * 2 * limit - limit;
                }
            }
        }
    }
}
