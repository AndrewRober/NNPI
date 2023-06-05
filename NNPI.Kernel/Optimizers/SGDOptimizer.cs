﻿using NNPI.Kernel.Optimizers.Base;

namespace NNPI.Kernel.Optimizers
{
    public class SGDOptimizer : OptimizerFunction
    {
        public SGDOptimizer(double learningRate) : base(learningRate)
        {
        }

        public override void UpdateWeights(double[] weights, double[] gradients)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= learningRate * gradients[i];
            }
        }
    }
}
