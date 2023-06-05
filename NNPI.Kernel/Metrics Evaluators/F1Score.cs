namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// F1Score metric class for evaluating binary classification models.
    /// </summary>
    public class F1Score : Metric
    {
        private Precision precisionMetric;
        private Recall recallMetric;

        /// <summary>
        /// Initializes a new instance of the F1Score class.
        /// </summary>
        public F1Score()
        {
            precisionMetric = new Precision();
            recallMetric = new Recall();
        }

        /// <summary>
        /// Computes the F1 score given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The F1 score value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match or when the labels are not binary.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            double precision = precisionMetric.Compute(trueLabels, predictedLabels);
            double recall = recallMetric.Compute(trueLabels, predictedLabels);

            return precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
        }
    }
}
