namespace NNPI.Kernel.Data_PreProcessors
{
    /// <summary>
    /// An advanced stemmer that implements the Porter Stemming Algorithm.
    /// </summary>
    public class AdvancedStemmer
    {
        private const string VowelPattern = "[aeiou]";
        private const string ConsonantPattern = "[^aeiou]";
        private const string CVPattern = ConsonantPattern + "*" + VowelPattern + ConsonantPattern;

        /// <summary>
        /// Stems a list of tokens using the Porter Stemming Algorithm.
        /// </summary>
        /// <param name="tokens">The input tokens (words) to be stemmed.</param>
        /// <returns>A list of stemmed tokens.</returns>
        public List<string> Stem(List<string> tokens)
        {
            var stemmedTokens = new List<string>();

            foreach (var token in tokens)
            {
                try
                {
                    stemmedTokens.Add(StemToken(token));
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error stemming token '{token}': {ex.Message}");
                    stemmedTokens.Add(token);
                }
            }

            return stemmedTokens;
        }

        /// <summary>
        /// Stems a single token using the Porter Stemming Algorithm.
        /// </summary>
        /// <param name="token">The input token (word) to be stemmed.</param>
        /// <returns>The stemmed token.</returns>
        private string StemToken(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                return token;
            }

            token = token.ToLowerInvariant();
            token = Step1A(token);
            token = Step1B(token);
            token = Step1C(token);
            token = Step2(token);
            token = Step3(token);
            token = Step4(token);
            token = Step5A(token);
            token = Step5B(token);

            return token;
        }

        // Implementation of the Porter Stemming Algorithm steps (1A to 5B)

        private string Step1A(string token)
        {
            // SSES -> SS
            if (token.EndsWith("sses") || token.EndsWith("ies"))
                return token[..^2];

            // SS -> SS
            if (token.EndsWith("ss"))
                return token;

            // S -> (empty)
            return token.EndsWith("s") ? token[..^1] : token;
        }

        private string Step1B(string token)
        {
            // (m > 0) EED -> EE
            if (token.EndsWith("eed") && GetM(token[..^3]) > 0)
                return token[..^1];

            // (*v*) ED -> (empty)
            if (token.EndsWith("ed") && ContainsVowel(token[..^2]))
                return Step1BHelper(token[..^2]);

            // (*v*) ING -> (empty)
            if (token.EndsWith("ing") && ContainsVowel(token[..^3]))
                return Step1BHelper(token[..^3]);

            return token;
        }

        private string Step1BHelper(string token)
        {
            if (token.EndsWith("at") || token.EndsWith("bl") || token.EndsWith("iz"))
                return token + "e";

            if (EndsWithDoubleConsonant(token) && !EndsWithCVC(token, "lsz"))
                return token[..^1];

            return GetM(token) == 1 && EndsWithCVC(token) ? token + "e" : token;
        }

        // (*v*) Y -> I
        private string Step1C(string token) =>
            token.EndsWith("y") && ContainsVowel(token[..^1]) ? token[..^1] + "i" : token;

        private string Step2(string token)
        {
            string[] suffixes =
            {
                "ational", "tional", "enci", "anci", "izer", "abli", "alli", "entli", "eli", "ousli", "ization",
                "ation", "ator", "alism", "iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "logi"
            };

            string[] replacements =
            {
                "ate", "tion", "ence", "ance", "ize", "able", "al", "ent", "e", "ous", "ize", "ate", "ate", "al",
                "ive", "ful", "ous", "al", "ive", "ble", "log"
            };

            for (int i = 0; i < suffixes.Length; i++)
            {
                if (token.EndsWith(suffixes[i]) && GetM(token[..^suffixes[i].Length]) > 0)
                    return token[..^suffixes[i].Length] + replacements[i];
            }

            return token;
        }

        private string Step3(string token)
        {
            string[] suffixes = { "icate", "ative", "alize", "iciti", "ical", "ful", "ness" };
            string[] replacements = { "ic", "", "al", "ic", "ic", "", "" };

            for (int i = 0; i < suffixes.Length; i++)
            {
                if (token.EndsWith(suffixes[i]) && GetM(token[..^suffixes[i].Length]) > 0)
                {
                    return token[..^suffixes[i].Length] + replacements[i];
                }
            }

            return token;
        }

        private string Step4(string token)
        {
            string[] suffixes =
            {
                "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ism", "ate", "iti",
                "ous", "ive", "ize"
            };

            for (int i = 0; i < suffixes.Length; i++)
            {
                if (token.EndsWith(suffixes[i]) && GetM(token[..^suffixes[i].Length]) > 1)
                {
                    return token[..^suffixes[i].Length];
                }
            }

            return token;
        }

        private string Step5A(string token)
        {
            // (m > 1) E -> (empty)
            if (token.EndsWith("e") && GetM(token[..^1]) > 1)
            {
                return token[..^1];
            }

            // (m = 1 and not *o) E -> (empty)
            return token.EndsWith("e") && GetM(token[..^1]) == 1 &&
                !EndsWithCVC(token[..^1]) ? token[..^1] : token;
        }

        // (m > 1 and *d and *L) -> (empty)
        private string Step5B(string token) => GetM(token) > 1 &&
            token.EndsWith("l") && token[token.Length - 2] == 'l' ?
                token[..^1] : token;

        // Helper methods for the Porter Stemming Algorithm

        private int GetM(string token)
        {
            int m = 0;
            int state = 0;

            for (int i = 0; i < token.Length; i++)
            {
                if (state == 0 && IsVowel(token[i]))
                {
                    state = 1;
                }
                else if (state == 1 && !IsVowel(token[i]))
                {
                    state = 0;
                    m++;
                }
            }

            return m;
        }

        private bool ContainsVowel(string token) => token.Any(IsVowel);

        private bool IsVowel(char c) => "aeiou".Contains(c);

        private bool EndsWithDoubleConsonant(string token) => token.Length < 2 ? false
            : token[token.Length - 1] == token[token.Length - 2]
                && !IsVowel(token[token.Length - 1]);

        private bool EndsWithCVC(string token, string exceptionChars = "") => token.Length < 3
                ? false
                : !IsVowel(token[token.Length - 1])
                    && IsVowel(token[token.Length - 2])
                    && !IsVowel(token[token.Length - 3])
                    && !exceptionChars.Contains(token[token.Length - 1]);
    }
}
