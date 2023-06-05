namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// GlorotUniformInitializer class that initializes the weights of a matrix with random values sampled from a uniform distribution
    /// with a range defined by the Glorot (Xavier) initialization method.
    /// </summary>
    public class GlorotUniformInitializer : WeightInitializer
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the GlorotUniformInitializer class.
        /// </summary>
        public GlorotUniformInitializer()
        {
            random = new Random();
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values sampled from a uniform distribution
        /// with a range defined by the Glorot (Xavier) initialization method.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);

            double limit = Math.Sqrt(6.0 / (inputSize + outputSize));

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
