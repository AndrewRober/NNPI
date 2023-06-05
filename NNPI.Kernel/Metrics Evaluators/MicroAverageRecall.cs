namespace NNPI.Kernel.Metrics_Evaluators
{
    public class MicroAverageRecall : Metric
    {
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            var labelSet = trueLabels.Concat(predictedLabels).Distinct().ToList();
            int tpSum = 0;
            int fnSum = 0;

            foreach (int label in labelSet)
            {
                int tp = 0;
                int fn = 0;

                for (int i = 0; i < trueLabels.Length; i++)
                {
                    if (trueLabels[i] == label)
                    {
                        if (predictedLabels[i] == label)
                        {
                            tp++;
                        }
                        else
                        {
                            fn++;
                        }
                    }
                }

                tpSum += tp;
                fnSum += fn;
            }

            if (tpSum + fnSum == 0)
            {
                return 0;
            }

            return (double)tpSum / (tpSum + fnSum);
        }
    }
}
