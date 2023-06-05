using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class NadamOptimizer : OptimizerFunction
    {
        private double[] m;
        private double[] v;
        private double beta1;
        private double beta2;
        private double epsilon;
        private int timestep;

        /// <summary>
        /// Creates a Nadam optimizer (Adam with Nesterov momentum).
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="beta1">The exponential decay rate for the first moment estimates.</param>
        /// <param name="beta2">The exponential decay rate for the second moment estimates.</param>
        /// <param name="epsilon">A small value to prevent division by zero.</param>
        public NadamOptimizer(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : base(learningRate)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.timestep = 0;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (m == null || v == null)
            {
                m = new double[weights.Length];
                v = new double[weights.Length];
            }

            timestep++;

            for (int i = 0; i < weights.Length; i++)
            {
                m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
                v[i] = beta2 * v[i] + (1 - beta2) * Math.Pow(gradients[i], 2);

                double mCorrected = m[i] / (1 - Math.Pow(beta1, timestep));
                double vCorrected = v[i] / (1 - Math.Pow(beta2, timestep));

                double mNesterov = mCorrected * (1 - Math.Pow(beta1, timestep)) / (1 - Math.Pow(beta1, timestep + 1));

                weights[i] -= learningRate * (mNesterov + (1 - beta1) * gradients[i]) / (Math.Sqrt(vCorrected) + epsilon);
            }
        }
    }
}
