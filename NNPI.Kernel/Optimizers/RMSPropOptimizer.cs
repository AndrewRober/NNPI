using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class RMSPropOptimizer : OptimizerFunction
    {
        private double[] meanOfSquaredGradients;
        private double decayRate;
        private double epsilon;

        /// <summary>
        /// Creates an RMSProp optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="decayRate">The decay rate for the moving average.</param>
        /// <param name="epsilon">A small value to prevent division by zero.</param>
        public RMSPropOptimizer(double learningRate, double decayRate = 0.9, double epsilon = 1e-8) : base(learningRate)
        {
            this.decayRate = decayRate;
            this.epsilon = epsilon;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (meanOfSquaredGradients == null)
            {
                meanOfSquaredGradients = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                meanOfSquaredGradients[i] = decayRate * meanOfSquaredGradients[i] + (1 - decayRate) * Math.Pow(gradients[i], 2);
                weights[i] -= learningRate * gradients[i] / (Math.Sqrt(meanOfSquaredGradients[i]) + epsilon);
            }
        }
    }
}
