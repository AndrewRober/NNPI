using NNPI.Kernel.Regularizers.Base;

namespace NNPI.Kernel.Regularizers
{
    /// <summary>
    /// L1L2 (Elastic Net) regularization implementation.
    /// </summary>
    public class L1L2 : RegularizerFunction
    {
        private double l1Ratio;

        public L1L2(double regularizationParameter, double l1Ratio) : base(regularizationParameter) => 
            this.l1Ratio = l1Ratio;

        /// <summary>
        /// Applies the regularization penalty to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to apply the regularization penalty to.</param>
        /// <returns>The regularization penalty.</returns>
        public override double Apply(double weight)
        {
            double l1Component = l1Ratio * Math.Abs(weight);
            double l2Component = (1 - l1Ratio) * 0.5 * Math.Pow(weight, 2);
            return regularizationParameter * (l1Component + l2Component);
        }

        /// <summary>
        /// Computes the gradient of the regularization penalty with respect to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to compute the gradient for.</param>
        /// <returns>The gradient of the regularization penalty.</returns>
        public override double Gradient(double weight)
        {
            double l1Component = l1Ratio * Math.Sign(weight);
            double l2Component = (1 - l1Ratio) * weight;
            return regularizationParameter * (l1Component + l2Component);
        }
    }
}
