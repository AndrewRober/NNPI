using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class GradientDescentOptimizer : OptimizerFunction
    {
        /// <summary>
        /// Creates a Gradient Descent optimizer.
        /// </summary>
        /// <param name="learningRate">The learning rate for the optimizer.</param>
        public GradientDescentOptimizer(double learningRate) : base(learningRate) { }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= learningRate * gradients[i];
            }
        }
    }
}
