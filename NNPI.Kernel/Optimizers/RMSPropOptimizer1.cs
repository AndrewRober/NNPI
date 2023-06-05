using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class RMSPropOptimizer : OptimizerFunction
    {
        private double[] _cache;
        private double _decayRate;
        private double _epsilon;

        public RMSPropOptimizer(double learningRate, double decayRate = 0.99, double epsilon = 1e-8) : base(learningRate)
        {
            this._decayRate = decayRate;
            this._epsilon = epsilon;
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            if (_cache == null)
            {
                _cache = new double[weights.Length];
            }

            for (int i = 0; i < weights.Length; i++)
            {
                _cache[i] = decayRate * _cache[i] + (1 - decayRate) * Math.Pow(gradients[i], 2);
                weights[i] -= learningRate * gradients[i] / (Math.Sqrt(_cache[i]) + epsilon);
            }
        }
    }
}
