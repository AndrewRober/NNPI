namespace NNPI.Kernel.EarlyStoppers
{
    /// <summary>
    /// RateOfChangeEarlyStopping class for monitoring training progress and stopping the training process
    /// when the rate of change in the monitored metric falls below a specified threshold.
    /// </summary>
    public class RateOfChangeEarlyStopping
    {
        private double rateOfChangeThreshold;
        private double? previousMetric;

        /// <summary>
        /// Initializes a new instance of the RateOfChangeEarlyStopping class with the given parameters.
        /// </summary>
        /// <param name="rateOfChangeThreshold">The rate of change threshold that triggers early stopping.</param>
        public RateOfChangeEarlyStopping(double rateOfChangeThreshold)
        {
            this.rateOfChangeThreshold = rateOfChangeThreshold;
            previousMetric = null;
        }

        /// <summary>
        /// Checks if early stopping should be triggered based on the current epoch's metric value.
        /// </summary>
        /// <param name="currentMetric">The metric value for the current epoch.</param>
        /// <returns>True if early stopping should be triggered, false otherwise.</returns>
        public bool ShouldStop(double currentMetric)
        {
            if (!previousMetric.HasValue)
            {
                previousMetric = currentMetric;
                return false;
            }

            double rateOfChange = (previousMetric.Value - currentMetric) / previousMetric.Value;
            previousMetric = currentMetric;

            return rateOfChange < rateOfChangeThreshold;
        }

        /// <summary>
        /// Resets the early stopping state, allowing it to be used for another training run.
        /// </summary>
        public void Reset()
        {
            previousMetric = null;
        }
    }
}
