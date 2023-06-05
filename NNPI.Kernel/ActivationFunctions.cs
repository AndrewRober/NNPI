namespace NNPI.Kernel
{
    /// <summary>
    /// A collection of common activation functions used in neural networks.
    /// This class contains the following activation functions:
    ///Sigmoid
    ///Hyperbolic Tangent (Tanh)
    ///Rectified Linear Unit (ReLU)
    ///Leaky Rectified Linear Unit (Leaky ReLU)
    ///Exponential Linear Unit (ELU)
    ///Softplus
    /// </summary>
    public static class ActivationFunctions
    {
        /// <summary>
        /// The Sigmoid activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <returns>The output value between 0 and 1.</returns>
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        /// <summary>
        /// The Hyperbolic Tangent (Tanh) activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <returns>The output value between -1 and 1.</returns>
        public static double Tanh(double x) => Math.Tanh(x);

        /// <summary>
        /// The Rectified Linear Unit (ReLU) activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <returns>The output value, which is the input value if it's positive, and 0 otherwise.</returns>
        public static double ReLU(double x) => Math.Max(0, x);

        /// <summary>
        /// The Leaky Rectified Linear Unit (Leaky ReLU) activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <param name="alpha">The slope of the function for negative input values.</param>
        /// <returns>The output value, which is the input value if it's positive, and alpha times the input value otherwise.</returns>
        public static double LeakyReLU(double x, double alpha = 0.01) => x > 0 ? x : alpha * x;

        /// <summary>
        /// The Exponential Linear Unit (ELU) activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <param name="alpha">The value to be multiplied with the exponent of the input value for negative input values.</param>
        /// <returns>The output value, which is the input value if it's positive, and alpha times (e^x - 1) otherwise.</returns>
        public static double ELU(double x, double alpha = 1.0) => x > 0 ? x : alpha * (Math.Exp(x) - 1);

        /// <summary>
        /// The Softplus activation function.
        /// </summary>
        /// <param name="x">The input value.</param>
        /// <returns>The output value, which is the natural logarithm of (1 + e^x).</returns>
        public static double Softplus(double x) => Math.Log(1 + Math.Exp(x));
    }
}