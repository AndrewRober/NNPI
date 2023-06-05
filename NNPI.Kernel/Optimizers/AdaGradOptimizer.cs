using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class AdaGradOptimizer : OptimizerFunction
    {
        private double[] sumOfSquaredGradients;
        private double epsilon;

        /// <summary>
        /// Creates an AdaGrad optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="epsilon">A small value to prevent division by zero.</param>
        public AdaGradOptimizer(double learningRate, double epsilon = 1e-8) : base(learningRate)
        {
            this.epsilon = epsilon;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (sumOfSquaredGradients == null)
                sumOfSquaredGradients = new double[weights.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                sumOfSquaredGradients[i] += Math.Pow(gradients[i], 2);
                weights[i] -= learningRate * gradients[i] / (Math.Sqrt(sumOfSquaredGradients[i]) + epsilon);
            }
        }
    }
}
