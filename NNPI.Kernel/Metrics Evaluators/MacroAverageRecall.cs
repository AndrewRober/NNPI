namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Macro-average recall metric class for evaluating multi-class classification models.
    /// </summary>
    public class MacroAverageRecall : Metric
    {
        private Recall recallMetric;

        /// <summary>
        /// Initializes a new instance of the MacroAverageRecall class.
        /// </summary>
        public MacroAverageRecall()
        {
            recallMetric = new Recall();
        }

        /// <summary>
        /// Computes the macro-average recall given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The macro-average recall value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            var labelSet = trueLabels.Concat(predictedLabels).Distinct().ToList();
            double sumRecall = 0.0;

            foreach (int label in labelSet)
            {
                sumRecall += recallMetric.Compute(trueLabels, predictedLabels, label);
            }

            return sumRecall / labelSet.Count;
        }
    }
}
