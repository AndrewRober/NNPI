using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class AdaMaxOptimizer : OptimizerFunction
    {
        private double[] m;
        private double[] u;
        private double beta1;
        private double beta2;
        private int timestep;

        /// <summary>
        /// Creates an AdaMax optimizer (variant of the Adam optimizer).
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="beta1">The exponential decay rate for the first moment estimates.</param>
        /// <param name="beta2">The exponential decay rate for the second moment estimates.</param>
        public AdaMaxOptimizer(double learningRate, double beta1 = 0.9, double beta2 = 0.999) : base(learningRate)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.timestep = 0;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (m == null || u == null)
            {
                m = new double[weights.Length];
                u = new double[weights.Length];
            }

            timestep++;

            for (int i = 0; i < weights.Length; i++)
            {
                m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
                u[i] = Math.Max(beta2 * u[i], Math.Abs(gradients[i]));

                double mCorrected = m[i] / (1 - Math.Pow(beta1, timestep));

                weights[i] -= learningRate * mCorrected / u[i];
            }
        }
    }
}
