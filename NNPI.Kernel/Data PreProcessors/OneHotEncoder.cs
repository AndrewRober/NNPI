namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Encodes categorical integer features as a one-hot array.
    /// </summary>
    public class OneHotEncoder
    {
        private int _numClasses;

        /// <summary>
        /// Initializes a new instance of the OneHotEncoder class with the specified number of classes.
        /// </summary>
        /// <param name="numClasses">The number of distinct classes in the input data.</param>
        public OneHotEncoder(int numClasses) => _numClasses = numClasses;

        /// <summary>
        /// Transforms the input data into a one-hot encoded array.
        /// </summary>
        /// <param name="data">A 1D array of input data.</param>
        /// <returns>A 2D array of one-hot encoded data.</returns>
        public int[][] Transform(int[] data)
        {
            int numRows = data.Length;

            int[][] oneHotData = new int[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                oneHotData[i] = new int[_numClasses];
                oneHotData[i][data[i]] = 1;
            }

            return oneHotData;
        }
    }
}
