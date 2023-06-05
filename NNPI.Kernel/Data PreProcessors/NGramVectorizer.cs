namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// A class that provides functionality to convert a collection of text documents into a matrix of n-gram counts.
    /// </summary>
    public class NGramVectorizer
    {
        private static readonly char[] delimiters = 
            new[] { ' ', ',', '.', ';', ':', '(', ')', '[', ']', '{', '}', '!', '?', '\'', '\"', '\\' };
        private List<string> _vocabulary;
        private int _n;

        /// <summary>
        /// Initializes a new instance of the NGramVectorizer class with a specified value for n and an optional vocabulary.
        /// </summary>
        /// <param name="n">The number of tokens to include in each n-gram.</param>
        /// <param name="vocabulary">An optional list of unique n-grams to be used as the vocabulary. If not provided, the vocabulary will be built from the input documents.</param>
        public NGramVectorizer(int n, List<string> vocabulary = null)
        {
            _n = n;
            _vocabulary = vocabulary;
        }

        /// <summary>
        /// Transforms a collection of text documents into a matrix of n-gram counts.
        /// </summary>
        /// <param name="documents">A collection of text documents to transform.</param>
        /// <returns>A matrix of n-gram counts, where each row represents a document and each column represents an n-gram in the vocabulary.</returns>
        public int[][] Transform(IEnumerable<string> documents)
        {
            if (_vocabulary == null)
                BuildVocabulary(documents);

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
            var ngramSet = new HashSet<string>();
            foreach (var ngram in documents.SelectMany(document => GenerateNGrams(Tokenize(document))))
                ngramSet.Add(ngram);

            _vocabulary = ngramSet.ToList();
        }

        /// <summary>
        /// Transforms a single document into an n-gram count vector.
        /// </summary>
        /// <param name="document">The text document to transform.</param>
        /// <returns>An n-gram count vector, where each element represents the count of an n-gram in the vocabulary.</returns>
        private int[] TransformDocument(string document)
        {
            var ngramCounts = new Dictionary<string, int>();

            string[] tokens = Tokenize(document);
            List<string> ngrams = GenerateNGrams(tokens);

            foreach (var ngram in ngrams)
            {
                if (ngramCounts.ContainsKey(ngram))
                {
                    ngramCounts[ngram]++;
                }
                else
                {
                    ngramCounts[ngram] = 1;
                }
            }

            int[] countVector = new int[_vocabulary.Count];

            for (int i = 0; i < _vocabulary.Count; i++)
                countVector[i] = ngramCounts.TryGetValue(_vocabulary[i], out int count) ? count : 0;

            return countVector;
        }

        /// <summary>
        /// Tokenizes the input document into an array of tokens.
        /// </summary>
        /// <param name="document">The text document to tokenize.</param>
        /// <returns>An array of tokens.</returns>
        private static string[] Tokenize(string document) =>
            document.ToLowerInvariant().Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

        /// <summary>
        /// Generates a list of n-grams from an array of tokens.
        /// </summary>
        /// <param name="tokens">An array of tokens to generate n-grams from.</param>
        /// <returns>A list of n-grams.</returns>
        private List<string> GenerateNGrams(string[] tokens)
        {
            List<string> ngrams = new List<string>();

            for (int i = 0; i <= tokens.Length - _n; i++)
                ngrams.Add(string.Join(" ", tokens.Skip(i).Take(_n)));

            return ngrams;
        }
    }
}
