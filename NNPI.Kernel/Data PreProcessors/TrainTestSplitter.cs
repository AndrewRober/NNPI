namespace NNPI.Kernel.Data_PreProcessors
{
    public class TrainTestSplitter
    {
        private double _testSize;
        private int _randomSeed;

        public TrainTestSplitter(double testSize = 0.25, int randomSeed = 0)
        {
            _testSize = testSize;
            _randomSeed = randomSeed;
        }

        public (T[][] train, T[][] test) Split<T>(T[][] data)
        {
            int numRows = data.Length;
            int numTestRows = (int)(numRows * _testSize);

            Random rng = new Random(_randomSeed);
            HashSet<int> testIndices = new HashSet<int>();

            while (testIndices.Count < numTestRows)
            {
                int randomIndex = rng.Next(numRows);
                testIndices.Add(randomIndex);
            }

            T[][] trainData = new T[numRows - numTestRows][];
            T[][] testData = new T[numTestRows][];

            for (int i = 0, trainIndex = 0, testIndex = 0; i < numRows; i++)
            {
                if (testIndices.Contains(i))
                {
                    testData[testIndex++] = data[i];
                }
                else
                {
                    trainData[trainIndex++] = data[i];
                }
            }

            return (trainData, testData);
        }
    }
}
