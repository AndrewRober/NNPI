namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// RandomNormalInitializer class that initializes the weights of a matrix with random values sampled from a normal (Gaussian) distribution.
    /// </summary>
    public class RandomNormalInitializer : WeightInitializer
    {
        private double mean;
        private double stddev;
        private Random random;

        /// <summary>
        /// Initializes a new instance of the RandomNormalInitializer class with the specified mean and standard deviation for the normal distribution.
        /// </summary>
        /// <param name="mean">The mean of the normal distribution. Default is 0.0.</param>
        /// <param name="stddev">The standard deviation of the normal distribution. Default is 0.1.</param>
        public RandomNormalInitializer(double mean = 0.0, double stddev = 0.1)
        {
            this.mean = mean;
            this.stddev = stddev;
            random = new Random();
        }

        /// <summary>
        /// Initializes the weights of a given matrix with random values sampled from a normal (Gaussian) distribution with the specified mean and standard deviation.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    weights[i, j] = SampleNormal(mean, stddev);
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
