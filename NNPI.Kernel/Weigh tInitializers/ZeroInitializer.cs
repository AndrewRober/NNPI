namespace NNPI.Kernel.Weigh_tInitializers
{
    /// <summary>
    /// ZeroInitializer class that initializes the weights of a matrix with zeros.
    /// </summary>
    public class ZeroInitializer : WeightInitializer
    {
        /// <summary>
        /// Initializes the weights of a given matrix with zeros.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public override void Initialize(double[,] weights)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    weights[i, j] = 0.0;
        }
    }
}
