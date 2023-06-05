using NNPI.Kernel.Regularizers.Base;

namespace NNPI.Kernel.Regularizers
{
    /// <summary>
    /// BatchNormalization class that implements batch normalization for feedforward neural networks.
    /// Inherits from the RegularizerFunction base class.
    /// </summary>
    public class BatchNormalization : RegularizerFunction
    {
        private double[] gamma;
        private double[] beta;
        private double epsilon;

        /// <summary>
        /// Initializes a new instance of the BatchNormalization class with the given input size and epsilon value.
        /// </summary>
        /// <param name="inputSize">The size of the input activations to normalize.</param>
        /// <param name="epsilon">A small constant for numerical stability. Default value is 1e-5.</param>
        public BatchNormalization(int inputSize, double epsilon = 1e-5) : base(0)
        {
            gamma = new double[inputSize];
            beta = new double[inputSize];
            this.epsilon = epsilon;

            for (int i = 0; i < inputSize; i++)
            {
                gamma[i] = 1.0;
                beta[i] = 0.0;
            }
        }

        /// <summary>
        /// Applies batch normalization to the given activation value using the calculated mean and variance.
        /// </summary>
        /// <param name="value">The activation value to normalize.</param>
        /// <returns>The normalized activation value.</returns>
        public override double Apply(double value)
        {
            // Batch normalization should be applied to a mini-batch of activations rather than a single activation value.
            // This method is provided for compatibility with the RegularizerFunction base class, but it is not recommended to use it.
            throw new NotImplementedException("BatchNormalization.Apply should be called for a mini-batch of activations, not a single value.");
        }

        /// <summary>
        /// Computes the gradient of the batch normalization function with respect to the given activation value.
        /// </summary>
        /// <param name="value">The activation value to compute the gradient for.</param>
        /// <returns>The gradient of the batch normalization function.</returns>
        public override double Gradient(double value)
        {
            // Batch normalization gradients should be computed for a mini-batch of activations rather than a single activation value.
            // This method is provided for compatibility with the RegularizerFunction base class, but it is not recommended to use it.
            throw new NotImplementedException("BatchNormalization.Gradient should be called for a mini-batch of activations, not a single value.");
        }

        /// <summary>
        /// Applies batch normalization to the input activations of a mini-batch using the calculated mean and variance.
        /// </summary>
        /// <param name="miniBatch">The mini-batch of input activations to normalize.</param>
        /// <returns>An array of normalized activations, averaged across the mini-batch.</returns>
        public double[] Apply(double[][] miniBatch)
        {
            int inputSize = gamma.Length;

            // Calculate the mean and variance for the mini-batch
            double[] mean = new double[inputSize];
            double[] variance = new double[inputSize];

            for (int i = 0; i < miniBatch.Length; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    mean[j] += miniBatch[i][j];
                }
            }

            for (int j = 0; j < inputSize; j++)
            {
                mean[j] /= miniBatch.Length;
            }

            for (int i = 0; i < miniBatch.Length; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    variance[j] += Math.Pow(miniBatch[i][j] - mean[j], 2);
                }
            }

            for (int j = 0; j < inputSize; j++)
            {
                variance[j] /= miniBatch.Length;
            }

            // Normalize the inputs using the calculated mean and variance
            double[][] normalizedBatch = new double[miniBatch.Length][];

            for (int i = 0; i < miniBatch.Length; i++)
            {
                normalizedBatch[i] = new double[inputSize];

                for (int j = 0; j < inputSize; j++)
                {
                    normalizedBatch[i][j] = (miniBatch[i][j] - mean[j]) / Math.Sqrt(variance[j] + epsilon);
                    normalizedBatch[i][j] = gamma[j] * normalizedBatch[i][j] + beta[j];
                }
            }

            // Compute the average normalized inputs to return
            double[] averageNormalizedInput = new double[inputSize];

            for (int i = 0; i < miniBatch.Length; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    averageNormalizedInput[j] += normalizedBatch[i][j];
                }
            }

            for (int j = 0; j < inputSize; j++)
            {
                averageNormalizedInput[j] /= miniBatch.Length;
            }

            return averageNormalizedInput;
        }
    }
}
