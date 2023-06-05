using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNPI.Kernel.Optimizers.Base
{
    public abstract class OptimizerFunction
    {
        protected double learningRate;

        public OptimizerFunction(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public abstract void UpdateWeights(double[] weights, double[] gradients);
    }
}
