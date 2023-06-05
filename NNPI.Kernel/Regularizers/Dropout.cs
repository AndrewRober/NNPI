using NNPI.Kernel.Regularizers.Base;

namespace NNPI.Kernel.Regularizers
{
    /// <summary>
    /// Dropout class that implements the dropout regularization technique for feedforward neural networks.
    /// Inherits from the RegularizerFunction base class.
    /// </summary>
    public class Dropout : RegularizerFunction
    {
        private Random random;

        /// <summary>
        /// Initializes a new instance of the Dropout class with the given dropout rate.
        /// </summary>
        /// <param name="dropoutRate">The dropout rate (between 0 and 1) specifying the probability of dropping a neuron's output during training. Default value is 0.5.</param>
        public Dropout(double dropoutRate = 0.5) : base(dropoutRate) => random = new Random();

        /// <summary>
        /// Applies dropout to the input activation by randomly zeroing the value.
        /// </summary>
        /// <param name="value">The input activation to apply dropout on.</param>
        /// <returns>The modified activation with dropout applied.</returns>
        public override double Apply(double value) => random.NextDouble() > regularizationParameter ? value : 0.0;

        /// <summary>
        /// Computes the gradient of the dropout function with respect to the given activation.
        /// </summary>
        /// <param name="value">The activation value to compute the gradient for.</param>
        /// <returns>The gradient of the dropout function, which is 1 if the value is not dropped and 0 if the value is dropped.</returns>
        public override double Gradient(double value) => value != 0.0 ? 1.0 : 0.0;
    }
}
