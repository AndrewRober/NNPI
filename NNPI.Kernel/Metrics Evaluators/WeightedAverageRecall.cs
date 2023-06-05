namespace NNPI.Kernel.Metrics_Evaluators
{
    public class WeightedAverageRecall : Metric
    {
        /// <summary>
        /// Computes the weighted average recall given the true labels and predicted labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>The weighted average recall value.</returns>
        /// <exception cref="ArgumentException">Thrown when the length of trueLabels and predictedLabels arrays do not match.</exception>
        public override double Compute(int[] trueLabels, int[] predictedLabels)
        {
            if (trueLabels.Length != predictedLabels.Length)
            {
                throw new ArgumentException("The length of trueLabels and predictedLabels arrays must be the same.");
            }

            Recall recall = new Recall();
            Dictionary<int, double> classWeights = GetClassWeights(trueLabels);
            Dictionary<int, Tuple<double, double>> perClassPrecisionRecall = CalculatePerClassPrecisionRecall(trueLabels, predictedLabels);

            double weightedAverageRecall = 0;
            foreach (var cls in perClassPrecisionRecall.Keys)
            {
                weightedAverageRecall += classWeights[cls] * perClassPrecisionRecall[cls].Item2;
            }

            return weightedAverageRecall;
        }

        /// <summary>
        /// Calculates the class weights based on the frequency of each class in the true labels.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <returns>A dictionary where the key is the class label, and the value is the class weight.</returns>
        private Dictionary<int, double> GetClassWeights(int[] trueLabels)
        {
            Dictionary<int, int> classCounts = new Dictionary<int, int>();
            int totalCount = trueLabels.Length;

            foreach (int label in trueLabels)
            {
                if (classCounts.ContainsKey(label))
                {
                    classCounts[label]++;
                }
                else
                {
                    classCounts[label] = 1;
                }
            }

            Dictionary<int, double> classWeights = new Dictionary<int, double>();
            foreach (var classCount in classCounts)
            {
                classWeights[classCount.Key] = (double)classCount.Value / totalCount;
            }

            return classWeights;
        }

        /// <summary>
        /// Calculates the per-class precision and recall values using the Precision and Recall classes.
        /// </summary>
        /// <param name="trueLabels">The true labels.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        /// <returns>A dictionary where the key is the class label, and the value is a tuple with precision and recall.</returns>
        protected Dictionary<int, Tuple<double, double>>
            CalculatePerClassPrecisionRecall(int[] trueLabels, int[] predictedLabels)
        {
            HashSet<int> distinctLabels = new HashSet<int>(trueLabels.Concat(predictedLabels));
            Dictionary<int, Tuple<double, double>> perClassPrecisionRecall = new Dictionary<int, Tuple<double, double>>();

            Precision precision = new Precision();
            Recall recall = new Recall();

            foreach (int label in distinctLabels)
            {
                double clsPrecision = precision.Compute(trueLabels, predictedLabels, label);
                double clsRecall = recall.Compute(trueLabels, predictedLabels, label);

                perClassPrecisionRecall[label] = Tuple.Create(clsPrecision, clsRecall);
            }

            return perClassPrecisionRecall;
        }
    }
}
