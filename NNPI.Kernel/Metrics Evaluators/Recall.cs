namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Recall metric class for evaluating binary classification models.
    /// </summary>
    public class Recall : Metric
    {
        /// <summary>
        /// Computes the recall given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The recall value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match or when the labels are not binary.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");

            int truePositives = 0;
            int falseNegatives = 0;

            for (int i = 0; i < trueLabels.Length; i++)
            {
                switch (trueLabels[i])
                {
                    case 1 when predictedLabels[i] == 1:
                        truePositives++;
                        break;
                    case 1 when predictedLabels[i] == 0:
                        falseNegatives++;
                        break;
                    default:
                        if (trueLabels[i] is not 0 and not 1)
                            throw new ArgumentException("The trueLabels array contains non-binary values.");
                        else if (predictedLabels[i] is not 0 and not 1)
                            throw new ArgumentException("The predictedLabels array contains non-binary values.");

                        break;
                }
            }

            return truePositives + falseNegatives == 0 ? 0 : (double)truePositives / (truePositives + falseNegatives);
        }
    }


}
