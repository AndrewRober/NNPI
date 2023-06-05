namespace NNPI.Kernel.EarlyStoppers
{
    /// <summary>
    /// TrainingLossEarlyStopping class for monitoring training progress and stopping the training process
    /// when the training loss exceeds a specified threshold.
    /// </summary>
    public class TrainingLossEarlyStopping
    {
        private double trainingLossThreshold;

        /// <summary>
        /// Initializes a new instance of the TrainingLossEarlyStopping class with the given parameters.
        /// </summary>
        /// <param name="trainingLossThreshold">The training loss threshold that triggers early stopping.</param>
        public TrainingLossEarlyStopping(double trainingLossThreshold)
        {
            this.trainingLossThreshold = trainingLossThreshold;
        }

        /// <summary>
        /// Checks if early stopping should be triggered based on the current epoch's training loss.
        /// </summary>
        /// <param name="currentTrainingLoss">The training loss for the current epoch.</param>
        /// <returns>True if early stopping should be triggered, false otherwise.</returns>
        public bool ShouldStop(double currentTrainingLoss)
        {
            return currentTrainingLoss > trainingLossThreshold;
        }
    }
}
