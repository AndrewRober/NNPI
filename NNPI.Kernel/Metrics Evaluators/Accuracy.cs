namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Accuracy metric class for evaluating classification models.
    /// </summary>
    public class Accuracy : Metric
    {
        /// <summary>
        /// Computes the accuracy given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The accuracy value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");

            int correctCount = 0;
            for (int i = 0; i < trueLabels.Length; i++)
                if (trueLabels[i] == predictedLabels[i])
                    correctCount++;

            return (double)correctCount / trueLabels.Length;
        }
    }


}
