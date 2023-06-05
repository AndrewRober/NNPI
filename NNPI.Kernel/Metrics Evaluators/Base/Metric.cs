namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Base class for evaluation metrics.
    /// </summary>
    public abstract class Metric
    {
        /// <summary>
        /// Computes the value of the metric given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The value of the metric.</returns>
        public abstract double Compute(int[] trueLabels, int[] predictedLabels);
    }
}
