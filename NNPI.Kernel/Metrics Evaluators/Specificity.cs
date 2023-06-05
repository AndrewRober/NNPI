namespace NNPI.Kernel.Metrics_Evaluators
{
    /// <summary>
    /// Specificity metric class for evaluating binary classification models.
    /// </summary>
    public class Specificity : Metric
    {
        /// <summary>
        /// Computes the specificity given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The specificity value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match or when the labels are not binary.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");

            int trueNegatives = 0;
            int falsePositives = 0;

            for (int i = 0; i < trueLabels.Length; i++)
            {
                switch (trueLabels[i])
                {
                    case 0 when predictedLabels[i] == 0:
                        trueNegatives++;
                        break;
                    case 0 when predictedLabels[i] == 1:
                        falsePositives++;
                        break;
                    default:
                        if (trueLabels[i] is not 0 and not 1)
                            throw new ArgumentException("The trueLabels array contains non-binary values.");
                        else if (predictedLabels[i] is not 0 and not 1)
                            throw new ArgumentException("The predictedLabels array contains non-binary values.");

                        break;
                }
            }

            return trueNegatives + falsePositives == 0 ? 0 : (double)trueNegatives / (trueNegatives + falsePositives);
        }
    }


}
