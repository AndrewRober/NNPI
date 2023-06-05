namespace NNPI.Kernel.Metrics_Evaluators
{
    public class MicroAveragePrecision : Metric
    {
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            var labelSet = trueLabels.Concat(predictedLabels).Distinct().ToList();
            int tpSum = 0;
            int fpSum = 0;

            foreach (int label in labelSet)
            {
                int tp = 0;
                int fp = 0;

                for (int i = 0; i < trueLabels.Length; i++)
                {
                    if (predictedLabels[i] == label)
                    {
                        if (trueLabels[i] == label)
                        {
                            tp++;
                        }
                        else
                        {
                            fp++;
                        }
                    }
                }

                tpSum += tp;
                fpSum += fp;
            }

            return tpSum + fpSum == 0 ? 0 : (double)tpSum / (tpSum + fpSum);
        }
    }
}
