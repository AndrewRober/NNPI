namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Balanced accuracy metric class for evaluating binary classification models.
    /// </summary>
    public class BalancedAccuracy : Metric
    {
        private Recall recallMetric;
        private Specificity specificityMetric;

        /// <summary>
        /// Initializes a new instance of the BalancedAccuracy class.
        /// </summary>
        public BalancedAccuracy()
        {
            recallMetric = new Recall();
            specificityMetric = new Specificity();
        }

        /// <summary>
        /// Computes the balanced accuracy given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The balanced accuracy value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            double sensitivity = recallMetric.Compute(trueLabels, predictedLabels);
            double specificity = specificityMetric.Compute(trueLabels, predictedLabels);

            return (sensitivity + specificity) / 2;
        }
    }
}
