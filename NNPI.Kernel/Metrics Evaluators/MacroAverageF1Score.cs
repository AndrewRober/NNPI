namespace NNPI.Kernel.Metrics_Evaluators
{
    public class MacroAverageF1Score : Metric
    {
        private MacroAveragePrecision macroAveragePrecision;
        private MacroAverageRecall macroAverageRecall;

        public MacroAverageF1Score()
        {
            macroAveragePrecision = new MacroAveragePrecision();
            macroAverageRecall = new MacroAverageRecall();
        }

        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");

            double precision = macroAveragePrecision.Compute(trueLabels, predictedLabels);
            double recall = macroAverageRecall.Compute(trueLabels, predictedLabels);

            return precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
        }
    }
}
