using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class AdagradOptimizer : OptimizerFunction
    {
        private double[] cache;
        private double epsilon;

        public AdagradOptimizer(double learningRate, double epsilon = 1e-8) : base(learningRate)
        {
            this.epsilon = epsilon;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (cache == null)
            {
                cache = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                cache[i] += Math.Pow(gradients[i], 2);
                weights[i] -= learningRate * gradients[i] / (Math.Sqrt(cache[i]) + epsilon);
            }
        }
    }
}
