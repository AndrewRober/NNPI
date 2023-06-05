namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Encodes string labels into integer labels.
    /// </summary>
    public class LabelEncoder
    {
        private Dictionary<string, int> _labelDictionary;

        /// <summary>
        /// Initializes a new instance of the LabelEncoder class.
        /// </summary>
        public LabelEncoder() => _labelDictionary = new Dictionary<string, int>();

        /// <summary>
        /// Fits the LabelEncoder to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 1D array of input data.</param>
        /// <returns>A 1D array of encoded integer labels.</returns>
        public int[] FitTransform(string[] data)
        {
            int numRows = data.Length;
            int[] encodedData = new int[numRows];
            int labelCounter = 0;

            for (int i = 0; i < numRows; i++)
            {
                string label = data[i];
                if (!_labelDictionary.ContainsKey(label))
                {
                    _labelDictionary[label] = labelCounter;
                    labelCounter++;
                }

                encodedData[i] = _labelDictionary[label];
            }

            return encodedData;
        }

        /// <summary>
        /// Transforms input data using the fitted LabelEncoder.
        /// </summary>
        /// <param name="data">A 1D array of input data.</param>
        /// <returns>A 1D array of encoded integer labels.</returns>
        public int[] Transform(string[] data)
        {
            int numRows = data.Length;
            int[] encodedData = new int[numRows];

            for (int i = 0; i < numRows; i++)
            {
                string label = data[i];
                encodedData[i] = _labelDictionary.ContainsKey(label)
                    ? _labelDictionary[label]
                    : throw new ArgumentException($"The label '{label}' was not found in the fitted LabelEncoder.");
            }

            return encodedData;
        }
    }
}
