namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Macro-average precision metric class for evaluating multi-class classification models.
    /// </summary>
    public class MacroAveragePrecision : Metric
    {
        private Precision precisionMetric;

        /// <summary>
        /// Initializes a new instance of the MacroAveragePrecision class.
        /// </summary>
        public MacroAveragePrecision()
        {
            precisionMetric = new Precision();
        }

        /// <summary>
        /// Computes the macro-average precision given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The macro-average precision value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            var labelSet = trueLabels.Concat(predictedLabels).Distinct().ToList();
            double sumPrecision = 0.0;

            foreach (int label in labelSet)
            {
                sumPrecision += precisionMetric.Compute(trueLabels, predictedLabels, label);
            }

            return sumPrecision / labelSet.Count;
        }
    }
}
