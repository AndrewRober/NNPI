namespace NNPI.Kernel.Regularizers.Base
{
    /// <summary>
    /// An abstract base class for implementing different types of regularizers.
    /// </summary>
    public abstract class RegularizerFunction
    {
        protected double regularizationParameter;

        public RegularizerFunction(double regularizationParameter) => 
            this.regularizationParameter = regularizationParameter;

        /// <summary>
        /// Applies the regularization penalty to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to apply the regularization penalty to.</param>
        /// <returns>The regularization penalty.</returns>
        public abstract double Apply(double weight);

        /// <summary>
        /// Computes the gradient of the regularization penalty with respect to the given weight.
        /// </summary>
        /// <param name="weight">The weight value to compute the gradient for.</param>
        /// <returns>The gradient of the regularization penalty.</returns>
        public abstract double Gradient(double weight);
    }
}
