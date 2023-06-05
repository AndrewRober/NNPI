namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Precision metric class for evaluating binary classification models.
    /// </summary>
    public class Precision : Metric
    {
        /// <summary>
        /// Computes the precision given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The precision value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match or when the labels are not binary.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            int truePositives = 0;
            int falsePositives = 0;

            for (int i = 0; i < trueLabels.Length; i++)
            {
                if (trueLabels[i] == 1 && predictedLabels[i] == 1)
                {
                    truePositives++;
                }
                else if (trueLabels[i] == 0 && predictedLabels[i] == 1)
                {
                    falsePositives++;
                }
                else if (trueLabels[i] != 0 && trueLabels[i] != 1)
                {
                    throw new ArgumentException("The trueLabels array contains non-binary values.");
                }
                else if (predictedLabels[i] != 0 && predictedLabels[i] != 1)
                {
                    throw new ArgumentException("The predictedLabels array contains non-binary values.");
                }
            }

            if (truePositives + falsePositives == 0)
            {
                return 0;
            }

            return (double)truePositives / (truePositives + falsePositives);
        }
    }
}
