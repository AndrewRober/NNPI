namespace NNPI.Kernel.Metrics_Evaluators
{
    public class ConfusionMatrix
    {
        public int[,] Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");

            var labelSet = trueLabels.Concat(predictedLabels).Distinct().ToList();
            int[,] confusionMatrix = new int[labelSet.Count, labelSet.Count];

            for (int i = 0; i < trueLabels.Length; i++)
            {
                int trueLabelIndex = labelSet.IndexOf(trueLabels[i]);
                int predictedLabelIndex = labelSet.IndexOf(predictedLabels[i]);

                confusionMatrix[trueLabelIndex, predictedLabelIndex]++;
            }

            return confusionMatrix;
        }
    }
}
