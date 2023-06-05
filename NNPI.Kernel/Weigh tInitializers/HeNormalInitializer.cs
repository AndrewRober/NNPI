namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// HeNormalInitializer class that initializes the weights of a matrix with random values sampled from a normal (Gaussian) distribution
    /// with parameters defined by the He (Kaiming) initialization method.
    /// </summary>
    public class HeNormalInitializer : WeightInitializer
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the HeNormalInitializer class.
        /// </summary>
        public HeNormalInitializer()
        {
            random = new Random();
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values sampled from a normal (Gaussian) distribution
        /// with parameters defined by the He (Kaiming) initialization method.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);

            double stddev = Math.Sqrt(2.0 / inputSize);

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weights[i, j] = SampleNormal(0.0, stddev);
                }
            }
        }

        /// <summary>
        /// Samples a random value from a normal (Gaussian) distribution with the specified mean and standard deviation using the Box-Muller transform.
        /// </summary>
        /// <param name="mean">The mean of the normal distribution.</param>
        /// <param name="stddev">The standard deviation of the normal distribution.</param>
        /// <returns>A random value sampled from the specified normal distribution.</returns>
        private double SampleNormal(double mean, double stddev)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();

            double z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return mean + stddev * z1;
        }
    }
}
