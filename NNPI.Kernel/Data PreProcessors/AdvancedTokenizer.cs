using System.Text.RegularExpressions;

namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// A more advanced tokenizer that supports different tokenization methods.
    /// </summary>
    public class AdvancedTokenizer
    {
        /// <summary>
        /// The type of tokenization to perform.
        /// </summary>
        public enum TokenizationType
        {
            Whitespace,
            Regex,
            NGrams
        }

        private TokenizationType _type;
        private int _n;
        private string _regexPattern;

        /// <summary>
        /// Initializes a new instance of the <see cref="AdvancedTokenizer"/> class.
        /// </summary>
        /// <param name="type">The type of tokenization to perform.</param>
        /// <param name="n">The number of characters in an n-gram. Only applicable for the NGrams method.</param>
        /// <param name="regexPattern">The regular expression pattern to use for tokenization. Only applicable for the Regex method.</param>
        public AdvancedTokenizer(TokenizationType type = TokenizationType.Whitespace, int n = 1, string regexPattern = @"\b[\w']+\b")
        {
            _type = type;
            _n = Math.Max(1, n);
            _regexPattern = regexPattern;
        }

        /// <summary>
        /// Tokenizes the input text using the specified tokenization method.
        /// </summary>
        /// <param name="text">The input text to be tokenized.</param>
        /// <returns>A list of tokens from the input text.</returns>
        public List<string> Tokenize(string text)
        {
            switch (_type)
            {
                case TokenizationType.Whitespace:
                    return WhitespaceTokenize(text);
                case TokenizationType.Regex:
                    return RegexTokenize(text);
                case TokenizationType.NGrams:
                    return NGramTokenize(text);
                default:
                    throw new ArgumentException("Invalid tokenization type.");
            }
        }

        /// <summary>
        /// Tokenizes the input text using whitespace as a delimiter.
        /// </summary>
        /// <param name="text">The input text to be tokenized.</param>
        /// <returns>A list of tokens from the input text.</returns>
        private List<string> WhitespaceTokenize(string text)
        {
            return new List<string>(text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries));
        }

        /// <summary>
        /// Tokenizes the input text using a regular expression pattern.
        /// </summary>
        /// <param name="text">The input text to be tokenized.</param>
        /// <returns>A list of tokens from the input text.</returns>
        private List<string> RegexTokenize(string text)
        {
            Regex regex = new Regex(_regexPattern);
            var tokens = new List<string>();

            foreach (Match match in regex.Matches(text))
            {
                tokens.Add(match.Value);
            }

            return tokens;
        }

        /// <summary>
        /// Tokenizes the input text into n-grams.
        /// </summary>
        /// <param name="text">The input text to be tokenized.</param>
        /// <returns>A list of n-grams from the input text.</returns>
        private List<string> NGramTokenize(string text)
        {
            var nGrams = new List<string>();

            for (int i = 0; i <= text.Length - _n; i++)
            {
                nGrams.Add(text.Substring(i, _n));
            }

            return nGrams;
        }
    }
}
