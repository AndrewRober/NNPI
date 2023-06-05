namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Binarizes labels using one-vs-all encoding.
    /// </summary>
    public class LabelBinarizer
    {
        private Dictionary<string, int> _labelToIndex;
        private Dictionary<int, string> _indexToLabel;

        /// <summary>
        /// Fits the binarizer to the input labels and transforms them into binary arrays.
        /// </summary>
        /// <param name="labels">An array of input labels.</param>
        /// <returns>A 2D array of binary arrays representing the input labels.</returns>
        public int[][] FitTransform(string[] labels)
        {
            if (labels == null || labels.Length == 0)
            {
                throw new ArgumentException("Labels must not be null or empty.", nameof(labels));
            }

            _labelToIndex = new Dictionary<string, int>();
            _indexToLabel = new Dictionary<int, string>();

            int numLabels = labels.Length;
            int numClasses = 0;

            foreach (string label in labels)
            {
                if (!_labelToIndex.ContainsKey(label))
                {
                    _labelToIndex[label] = numClasses;
                    _indexToLabel[numClasses] = label;
                    numClasses++;
                }
            }

            int[][] binaryArrays = new int[numLabels][];
            for (int i = 0; i < numLabels; i++)
            {
                binaryArrays[i] = new int[numClasses];
                binaryArrays[i][_labelToIndex[labels[i]]] = 1;
            }

            return binaryArrays;
        }

        /// <summary>
        /// Inverts the binary arrays back into their original labels.
        /// </summary>
        /// <param name="binaryArrays">A 2D array of binary arrays.</param>
        /// <returns>An array of original labels.</returns>
        public string[] InverseTransform(int[][] binaryArrays)
        {
            if (binaryArrays == null || binaryArrays.Length == 0)
            {
                throw new ArgumentException("Binary arrays must not be null or empty.", nameof(binaryArrays));
            }

            int numLabels = binaryArrays.Length;
            string[] labels = new string[numLabels];

            for (int i = 0; i < numLabels; i++)
            {
                int index = Array.IndexOf(binaryArrays[i], 1);
                labels[i] = _indexToLabel[index];
            }

            return labels;
        }
    }

}
