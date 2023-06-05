using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNPI.Kernel.Loss
{
    internal class LossFunctions
    {
        /// <summary>
        /// Calculates the Mean Squared Error (MSE) loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <returns>The MSE loss.</returns>
        public static double MeanSquaredError(double[] predictions, double[] targets) =>
            predictions.Zip(targets, (p, t) => Math.Pow(p - t, 2)).Average();

        /// <summary>
        /// Calculates the Mean Absolute Error (MAE) loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <returns>The MAE loss.</returns>
        public static double MeanAbsoluteError(double[] predictions, double[] targets) =>
            predictions.Zip(targets, (p, t) => Math.Abs(p - t)).Average();

        /// <summary>
        /// Calculates the Root Mean Squared Error (RMSE) loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <returns>The RMSE loss.</returns>
        public static double RootMeanSquaredError(double[] predictions, double[] targets) =>
            Math.Sqrt(MeanSquaredError(predictions, targets));

        /// <summary>
        /// Calculates the Cross-Entropy loss for binary classification.
        /// </summary>
        /// <param name="predictions">Predicted probability values.</param>
        /// <param name="targets">Actual binary values (0 or 1).</param>
        /// <returns>The Cross-Entropy loss.</returns>
        public static double BinaryCrossEntropy(double[] predictions, double[] targets) =>
            -1 * targets.Zip(predictions, (t, p) =>
                                        t * Math.Log(p) + (1 - t) * Math.Log(1 - p)).Average();

        /// <summary>
        /// Calculates the Hinge loss for binary classification.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual binary values (-1 or 1).</param>
        /// <returns>The Hinge loss.</returns>
        public static double HingeLoss(double[] predictions, double[] targets) =>
            predictions.Zip(targets, (p, t) => Math.Max(0, 1 - t * p)).Average();

        /// <summary>
        /// Calculates the Huber loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <param name="delta">The delta parameter for the Huber loss.</param>
        /// <returns>The Huber loss.</returns>
        public static double HuberLoss(double[] predictions, double[] targets, double delta = 1.0) =>
            predictions.Zip(targets, (p, t) =>
                {
                    double diff = Math.Abs(p - t);
                    return diff <= delta ? 0.5 * Math.Pow(diff, 2) : delta * (diff - 0.5 * delta);
                }).Average();

        /// <summary>
        /// Calculates the Log-Cosh loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <returns>The Log-Cosh loss.</returns>
        public static double LogCoshLoss(double[] predictions, double[] targets) =>
            predictions.Zip(targets, (p, t) => Math.Cosh(p - t) - 1).Average();

        /// <summary>
        /// Calculates the Quantile loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <param name="quantile">The quantile parameter (between 0 and 1).</param>
        /// <returns>The Quantile loss.</returns>
        public static double QuantileLoss(double[] predictions, double[] targets, double quantile = 0.5) =>
            predictions.Zip(targets, (p, t) =>
                                        t >= p ? quantile * (t - p) : (1 - quantile) * (p - t)).Average();

        /// <summary>
        /// Calculates the Poisson loss.
        /// </summary>
        /// <param name="predictions">Predicted values.</param>
        /// <param name="targets">Actual values.</param>
        /// <returns>The Poisson loss.</returns>
        public static double PoissonLoss(double[] predictions, double[] targets) =>
            predictions.Zip(targets, (p, t) => p - t * Math.Log(p)).Average();

        /// <summary>
        /// Calculates the Kullback-Leibler Divergence loss.
        /// </summary>
        /// <param name="predictions">Predicted probability values.</param>
        /// <param name="targets">Actual probability values.</param>
        /// <returns>The Kullback-Leibler Divergence loss.</returns>
        public static double KullbackLeiblerDivergence(double[] predictions, double[] targets) =>
            targets.Zip(predictions, (t, p) => t * Math.Log(t / p)).Sum();

        /// <summary>
        /// Calculates the Cosine Similarity loss.
        /// </summary>
        /// <param name="predictions">Predicted vector values.</param>
        /// <param name="targets">Actual vector values.</param>
        /// <returns>The Cosine Similarity loss.</returns>
        public static double CosineSimilarityLoss(double[] predictions, double[] targets)
        {
            double dotProduct = predictions.Zip(targets, (p, t) => p * t).Sum();
            double predNorm = Math.Sqrt(predictions.Sum(p => Math.Pow(p, 2)));
            double targetNorm = Math.Sqrt(targets.Sum(t => Math.Pow(t, 2)));
            return 1 - dotProduct / (predNorm * targetNorm);
        }

        /// <summary>
        /// Calculates the Focal loss for binary classification.
        /// </summary>
        /// <param name="predictions">Predicted probability values.</param>
        /// <param name="targets">Actual binary values (0 or 1).</param>
        /// <param name="alpha">A balancing factor (typically between 0 and 1).</param>
        /// <param name="gamma">A focusing factor (typically non-negative).</param>
        /// <returns>The Focal loss.</returns>
        public static double FocalLoss(double[] predictions, double[] targets, double alpha = 0.25, double gamma = 2.0)
        {
            return -targets.Zip(predictions, (t, p) => t * alpha * Math.Pow(1 - p, gamma) * Math.Log(p) + (1 - t) * (1 - alpha) * Math.Pow(p, gamma) * Math.Log(1 - p)).Average();
        }

        /// <summary>
        /// Calculates the Dice loss.
        /// </summary>
        /// <param name="predictions">Predicted binary values (0 or 1).</param>
        /// <param name="targets">Actual binary values (0 or 1).</param>
        /// <returns>The Dice loss.</returns>
        public static double DiceLoss(double[] predictions, double[] targets)
        {
            double intersection = predictions.Zip(targets, (p, t) => p * t).Sum();
            double sum = predictions.Sum() + targets.Sum();
            return 1 - 2 * intersection / sum;
        }

        /// <summary>
        /// Calculates the Tversky loss.
        /// </summary>
        /// <param name="predictions">Predicted binary values (0 or 1).</param>
        /// <param name="targets">Actual binary values (0 or 1).</param>
        /// <param name="alpha">A weighting factor for false positives (typically between 0 and 1).</param>
        /// <param name="beta">A weighting factor for false negatives (typically between 0 and 1).</param>
        /// <returns>The Tversky loss.</returns>
        public static double TverskyLoss(double[] predictions, double[] targets, double alpha = 0.5, double beta = 0.5)
        {
            double tp = predictions.Zip(targets, (p, t) => p * t).Sum();
            double fp = predictions.Zip(targets, (p, t) => (1 - t) * p).Sum();
            double fn = predictions.Zip(targets, (p, t) => t * (1 - p)).Sum();
            return 1 - tp / (tp + alpha * fp + beta * fn);
        }

        /// <summary>
        /// Calculates the Categorical Cross-Entropy loss.
        /// </summary>
        /// <param name="predictions">Predicted probability values as a jagged array (rows: samples, columns: classes).</param>
        /// <param name="targets">Actual one-hot encoded values as a jagged array (rows: samples, columns: classes).</param>
        /// <returns>The Categorical Cross-Entropy loss.</returns>
        public static double CategoricalCrossEntropy(double[][] predictions, double[][] targets) =>
            -1 * Enumerable.Range(0, predictions.Length)
            .Select(i => targets[i].Zip(predictions[i],
                (t, p) => t * Math.Log(p)).Sum()).Average();

        /// <summary>
        /// Calculates the Sparse Categorical Cross-Entropy loss.
        /// </summary>
        /// <param name="predictions">Predicted probability values as a jagged array (rows: samples, columns: classes).</param>
        /// <param name="targets">Actual class indices as an array of integers.</param>
        /// <returns>The Sparse Categorical Cross-Entropy loss.</returns>
        public static double SparseCategoricalCrossEntropy(double[][] predictions, int[] targets) =>
            -1 * Enumerable.Range(0, predictions.Length)
            .Select(i => Math.Log(predictions[i][targets[i]])).Average();
    }
}
