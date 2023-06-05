using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class MomentumOptimizer : OptimizerFunction
    {
        private double[] velocity;
        private double momentum;

        /// <summary>
        /// Creates a Momentum optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        /// <param name="momentum">The momentum factor for the optimizer.</param>
        public MomentumOptimizer(double learningRate, double momentum = 0.9) : base(learningRate)
        {
            this.momentum = momentum;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (velocity == null)
            {
                velocity = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                velocity[i] = momentum * velocity[i] - learningRate * gradients[i];
                weights[i] += velocity[i];
            }
        }
    }
}
