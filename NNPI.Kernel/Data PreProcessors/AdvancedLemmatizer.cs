namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// A class that provides advanced lemmatization capabilities using preloaded lemma/token pairs from a text file.
    /// </summary>
    public class Lemmatizer
    {
        private static Dictionary<string, string> _lemmaDictionary;

        /// <summary>
        /// Static constructor that initializes the LemmaDictionary by loading lemma/token pairs from the specified file.
        /// </summary>
        static Lemmatizer() => LoadLemmatizationData("res/lemma/lemmatization-en.txt");

        /// <summary>
        /// Loads lemmatization data from the specified file.
        /// </summary>
        /// <param name="filePath">The path to the file containing lemma/token pairs.</param>
        private static void LoadLemmatizationData(string filePath)
        {
            try
            {
                // Initialize the lemma dictionary
                _lemmaDictionary = new Dictionary<string, string>();

                // Read the file line by line
                using (StreamReader reader = new StreamReader(filePath))
                {
                    string line;

                    while ((line = reader.ReadLine()) != null)
                    {
                        // Split the line into lemma and token
                        string[] lemmaTokenPair = line.Trim().Split('\t');

                        // Add the lemma/token pair to the dictionary
                        if (lemmaTokenPair.Length == 2)
                        {
                            _lemmaDictionary[lemmaTokenPair[1]] = lemmaTokenPair[0];
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error loading lemmatization data: {ex.Message}");
            }
        }

        /// <summary>
        /// Lemmatizes a list of tokens using the loaded lemma/token pairs.
        /// </summary>
        /// <param name="tokens">A list of tokens to lemmatize.</param>
        /// <returns>A list of lemmatized tokens.</returns>
        public List<string> Lemmatize(List<string> tokens)
        {
            var lemmatizedTokens = new List<string>();

            foreach (var token in tokens)
            {
                try
                {
                    // Lemmatize the token and add it to the lemmatized tokens list
                    string lemma = LemmatizeToken(token);
                    lemmatizedTokens.Add(lemma);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error lemmatizing token '{token}': {ex.Message}");
                    lemmatizedTokens.Add(token);
                }
            }

            return lemmatizedTokens;
        }

        /// <summary>
        /// Lemmatizes a single token using the loaded lemma/token pairs.
        /// </summary>
        /// <param name="token">The token to lemmatize.</param>
        /// <returns>The lemmatized form of the token, or the original token if no lemma is found.</returns>
        private static string LemmatizeToken(string token)
        {
            if (string.IsNullOrEmpty(token))
                return token;

            // Try to find the lemma for the token in the dictionary
            if (_lemmaDictionary.TryGetValue(token, out string lemma))
                return lemma;

            // Return the original token if no lemma is found
            return token;
        }
    }
}
