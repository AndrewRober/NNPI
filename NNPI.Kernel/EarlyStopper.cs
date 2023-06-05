using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNPI.Kernel
{
    /// <summary>
    /// EarlyStopping class for monitoring training progress and stopping the training process
    /// when the monitored metric stops improving for a specified number of consecutive epochs.
    /// </summary>
    public class EarlyStopper
    {
        private int patience;
        private double minDelta;
        private int epochsWithoutImprovement;
        private double bestMetric;

        /// <summary>
        /// Initializes a new instance of the EarlyStopping class with the given parameters.
        /// </summary>
        /// <param name="patience">The number of consecutive epochs without improvement required to trigger early stopping.</param>
        /// <param name="minDelta">The minimum change in the monitored metric required to qualify as an improvement.</param>
        public EarlyStopper(int patience, double minDelta = 0)
        {
            this.patience = patience;
            this.minDelta = minDelta;
            epochsWithoutImprovement = 0;
            bestMetric = double.MaxValue;
        }

        /// <summary>
        /// Checks if early stopping should be triggered based on the current epoch's metric value.
        /// </summary>
        /// <param name="currentMetric">The metric value for the current epoch.</param>
        /// <returns>True if early stopping should be triggered, false otherwise.</returns>
        public bool ShouldStop(double currentMetric)
        {
            if (bestMetric - currentMetric > minDelta)
            {
                epochsWithoutImprovement = 0;
                bestMetric = currentMetric;
            }
            else
            {
                epochsWithoutImprovement++;
            }

            return epochsWithoutImprovement >= patience;
        }

        /// <summary>
        /// Resets the early stopping state, allowing it to be used for another training run.
        /// </summary>
        public void Reset()
        {
            epochsWithoutImprovement = 0;
            bestMetric = double.MaxValue;
        }
    }
}
