namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// Scales the input data to a specified range (default: [0, 1]) using Min-Max scaling.
    /// </summary>
    public class MinMaxScaler
    {
        private double _min;
        private double _max;
        private double _dataMin;
        private double _dataMax;

        /// <summary>
        /// Initializes a new instance of the MinMaxScaler class with the specified min and max values.
        /// </summary>
        /// <param name="min">The minimum value of the desired output range.</param>
        /// <param name="max">The maximum value of the desired output range.</param>
        public MinMaxScaler(double min = 0, double max = 1)
        {
            _min = min;
            _max = max;
        }

        /// <summary>
        /// Fits the MinMaxScaler to the input data and transforms it.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        /// <returns>A 2D array of normalized data.</returns>
        public double[][] FitTransform(double[][] data)
        {
            int numRows = data.Length;
            int numCols = data[0].Length;

            double[][] normalizedData = new double[numRows][];
            for (int i = 0; i < numRows; i++)
                normalizedData[i] = new double[numCols];

            for (int col = 0; col < numCols; col++)
            {
                _dataMin = double.MaxValue;
                _dataMax = double.MinValue;

                // Find the min and max values in each column
                for (int row = 0; row < numRows; row++)
                {
                    _dataMin = Math.Min(_dataMin, data[row][col]);
                    _dataMax = Math.Max(_dataMax, data[row][col]);
                }

                // Normalize the data in each column
                for (int row = 0; row < numRows; row++)
                    normalizedData[row][col] = ((_max - _min) * (data[row][col] - _dataMin) / (_dataMax - _dataMin)) + _min;
            }

            return normalizedData;
        }
    }


}
