namespace NNPI.Kernel.Metrics_Evaluators
{
    public class WeightedAverageF1Score : Metric
    {
        private WeightedAveragePrecision weightedAveragePrecision;
        private WeightedAverageRecall weightedAverageRecall;

        public WeightedAverageF1Score()
        {
            weightedAveragePrecision = new WeightedAveragePrecision();
            weightedAverageRecall = new WeightedAverageRecall();
        }

        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            double precision = weightedAveragePrecision.Compute(trueLabels, predictedLabels);
            double recall = weightedAverageRecall.Compute(trueLabels, predictedLabels);

            return precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
        }
    }
}
