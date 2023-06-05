using NNPI.Kernel.Regularizers.Base;

namespace NNPI.Kernel.Regularizers
{
    /// <summary>
    /// L2 (Ridge) regularization implementation.
    /// </summary>
    public class L2 : RegularizerFunction
    {
        public L2(double regularizationParameter) : base(regularizationParameter)
        {
        }

        /// <summary>
        /// Applies the regularization penalty to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to apply the regularization penalty to.</param>
        /// <returns>The regularization penalty.</returns>
        public override double Apply(double weight) => 0.5 * regularizationParameter * Math.Pow(weight, 2);

        /// <summary>
        /// Computes the gradient of the regularization penalty with respect to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to compute the gradient for.</param>
        /// <returns>The gradient of the regularization penalty.</returns>
        public override double Gradient(double weight) => regularizationParameter * weight;
    }
}
