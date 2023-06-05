using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class RPropOptimizer : OptimizerFunction
    {
        private double[] prevGradients;
        private double[] stepSizes;
        private double etaPlus;
        private double etaMinus;
        private double stepSizeMin;
        private double stepSizeMax;

        public RPropOptimizer(double learningRate, double etaPlus = 1.2, double etaMinus = 0.5, double stepSizeMin = 1e-6, double stepSizeMax = 50.0)
            : base(learningRate)
        {
            this.etaPlus = etaPlus;
            this.etaMinus = etaMinus;
            this.stepSizeMin = stepSizeMin;
            this.stepSizeMax = stepSizeMax;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (prevGradients == null || stepSizes == null)
            {
                prevGradients = new double[weights.Length];
                stepSizes = new double[weights.Length];
                for (int i = 0; i < stepSizes.Length; i++)
                {
                    stepSizes[i] = learningRate;
                }
            }

            for (int i = 0; i < weights.Length; i++)
            {
                double gradientSignChange = prevGradients[i] * gradients[i];

                if (gradientSignChange > 0)
                {
                    stepSizes[i] = Math.Min(stepSizes[i] * etaPlus, stepSizeMax);
                    weights[i] -= Math.Sign(gradients[i]) * stepSizes[i];
                    prevGradients[i] = gradients[i];
                }
                else if (gradientSignChange < 0)
                {
                    stepSizes[i] = Math.Max(stepSizes[i] * etaMinus, stepSizeMin);
                    prevGradients[i] = 0;
                }
                else
                {
                    weights[i] -= Math.Sign(gradients[i]) * stepSizes[i];
                    prevGradients[i] = gradients[i];
                }
            }
        }
    }
}
