namespace NNPI.Kernel.Metrics_Evaluators
{
    public class MicroAverageF1Score : Metric
    {
        private MicroAveragePrecision microAveragePrecision;
        private MicroAverageRecall microAverageRecall;

        public MicroAverageF1Score()
        {
            microAveragePrecision = new MicroAveragePrecision();
            microAverageRecall = new MicroAverageRecall();
        }

        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            double precision = microAveragePrecision.Compute(trueLabels, predictedLabels);
            double recall = microAverageRecall.Compute(trueLabels, predictedLabels);

            if (precision + recall == 0)
            {
                return 0;
            }

            return 2 * (precision * recall) / (precision + recall);
        }
    }
}
