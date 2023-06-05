using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class AdaDeltaOptimizer : OptimizerFunction
    {
        private double[] accumulateGrad;
        private double[] accumulateUpdate;
        private double rho;
        private double epsilon;

        /// <summary>
        /// Creates an AdaDelta optimizer.
        /// </summary>
        /// <param name="rho">The decay rate for the moving averages of the squared gradients and updates.</param>
        /// <param name="epsilon">A small value to prevent division by zero.</param>
        public AdaDeltaOptimizer(double rho = 0.95, double epsilon = 1e-6) : base(0.0)
        {
            this.rho = rho;
            this.epsilon = epsilon;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (accumulateGrad == null || accumulateUpdate == null)
            {
                accumulateGrad = new double[weights.Length];
                accumulateUpdate = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                accumulateGrad[i] = rho * accumulateGrad[i] + (1 - rho) * Math.Pow(gradients[i], 2);
                double delta = -Math.Sqrt((accumulateUpdate[i] + epsilon) / (accumulateGrad[i] + epsilon)) * gradients[i];
                accumulateUpdate[i] = rho * accumulateUpdate[i] + (1 - rho) * Math.Pow(delta, 2);
                weights[i] += delta;
            }
        }
    }
}
