namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Imputes missing values in the input data using the most frequent value of the non-missing values in the same column.
    /// </summary>
    public class MostFrequentImputer
    {
        private object[] _mostFrequent;

        /// <summary>
        /// Fits the MostFrequentImputer to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of imputed data.</returns>
        public object[][] FitTransform(object[][] data)
        {
            int numRows = data.Length;
            int numCols = data[0].Length;

            _mostFrequent = new object[numCols];

            object[][] imputedData = new object[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                imputedData[i] = new object[numCols];
            }

            // Calculate the most frequent value for each column
            for (int col = 0; col < numCols; col++)
            {
                Dictionary<object, int> valueCounts = new Dictionary<object, int>();

                for (int row = 0; row < numRows; row++)
                {
                    object value = data[row][col];
                    if (value != null)
                    {
                        if (valueCounts.ContainsKey(value))
                        {
                            valueCounts[value]++;
                        }
                        else
                        {
                            valueCounts[value] = 1;
                        }
                    }
                }

                _mostFrequent[col] = valueCounts.OrderByDescending(x => x.Value).First().Key;
            }

            // Impute missing values using the column most frequent value
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    imputedData[row][col] = data[row][col] ?? _mostFrequent[col];
                }
            }

            return imputedData;
        }
    }
}
