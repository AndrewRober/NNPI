namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// A class that provides functionality to convert a collection of text documents into a matrix of token counts.
    /// </summary>
    public class CountVectorizer
    {
        private List<string> _vocabulary;
        static readonly char[] delimiters = new[] { ' ', ',', '.', ';', ':', '(', ')', '[', ']', '{', '}', '!', '?', '\'', '\"', '\\' };

        /// <summary>
        /// Initializes a new instance of the CountVectorizer class with an optional vocabulary.
        /// </summary>
        /// <param name="vocabulary">An optional list of unique tokens to be used as the vocabulary. If not provided, the vocabulary will be built from the input documents.</param>
        public CountVectorizer(List<string> vocabulary = null) => _vocabulary = vocabulary;

        /// <summary>
        /// Transforms a collection of text documents into a matrix of token counts.
        /// </summary>
        /// <param name="documents">A collection of text documents to transform.</param>
        /// <returns>A matrix of token counts, where each row represents a document and each column represents a token in the vocabulary.</returns>
        public int[][] Transform(IEnumerable<string> documents)
        {
            if (_vocabulary == null)
            {
                BuildVocabulary(documents);
            }

            int[][] countMatrix = new int[documents.Count()][];

            int docIndex = 0;
            foreach (var document in documents)
            {
                countMatrix[docIndex] = TransformDocument(document);
                docIndex++;
            }

            return countMatrix;
        }

        /// <summary>
        /// Builds the vocabulary from the input documents.
        /// </summary>
        /// <param name="documents">A collection of text documents to build the vocabulary from.</param>
        private void BuildVocabulary(IEnumerable<string> documents)
        {
            var tokenSet = new HashSet<string>();

            foreach (var document in documents)
                foreach (var token in Tokenize(document))
                    tokenSet.Add(token);

            _vocabulary = tokenSet.ToList();
        }

        /// <summary>
        /// Transforms a single document into a token count vector.
        /// </summary>
        /// <param name="document">The text document to transform.</param>
        /// <returns>A token count vector, where each element represents the count of a token in the vocabulary.</returns>
        private int[] TransformDocument(string document)
        {
            var tokenCounts = new Dictionary<string, int>();

            string[] tokens = Tokenize(document);

            foreach (var token in tokens)
            {
                if (tokenCounts.ContainsKey(token))
                {
                    tokenCounts[token]++;
                }
                else
                {
                    tokenCounts[token] = 1;
                }
            }

            int[] countVector = new int[_vocabulary.Count];

            for (int i = 0; i < _vocabulary.Count; i++)
                countVector[i] = tokenCounts.TryGetValue(_vocabulary[i], out int count) ? count : 0;

            return countVector;
        }

        /// <summary>
        /// Tokenizes the input document into an array of tokens.
        /// </summary>
        /// <param name="document">The text document to tokenize.</param>
        /// <returns>An array of tokens.</returns>
        private static string[] Tokenize(string document) =>
            document.ToLowerInvariant().Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
    }
}
