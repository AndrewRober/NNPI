using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class FTRLOptimizer : OptimizerFunction
    {
        private double[] z;
        private double[] n;
        private double alpha;
        private double beta;
        private double l1;
        private double l2;

        /// <summary>
        /// Creates a FTRL (Follow The Regularized Leader) optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="alpha">Hyperparameter that determines the strength of the adaptive learning rate.</param>
        /// <param name="beta">Hyperparameter that helps with initialization of the adaptive learning rate.</param>
        /// <param name="l1">The L1 regularization strength.</param>
        /// <param name="l2">The L2 regularization strength.</param>
        public FTRLOptimizer(double learningRate, double alpha = 0.005, double beta = 1.0, double l1 = 1.0, double l2 = 1.0) : base(learningRate)
        {
            this.alpha = alpha;
            this.beta = beta;
            this.l1 = l1;
            this.l2 = l2;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (z == null || n == null)
            {
                z = new double[weights.Length];
                n = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                double g = gradients[i];
                double sigma = (Math.Sqrt(n[i] + g * g) - Math.Sqrt(n[i])) / alpha;
                z[i] += g - sigma * weights[i];
                n[i] += g * g;

                if (Math.Abs(z[i]) <= l1)
                {
                    weights[i] = 0;
                }
                else
                {
                    double sign = z[i] >= 0 ? 1 : -1;
                    weights[i] = -(z[i] - sign * l1) / ((beta + Math.Sqrt(n[i])) / alpha + l2);
                }
            }
        }
    }
}
