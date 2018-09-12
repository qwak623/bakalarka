using System.Collections.Generic;

namespace AnyFileRNN
{
    public class Vocabulary
    {
        SortedDictionary<string, int> dictionary = new SortedDictionary<string, int>();
        List<string> list = new List<string>();
        const string sentenceStart = "_START_";
        const string sentenceEnd = "_END_";
        const string unknownToken = "_UNKNOWN_";

        public int Size { get { return list.Count + 1; } }

        public Vocabulary(IEnumerable<string> words)
        {
            dictionary.Add(sentenceStart, dictionary.Count);
            list.Add(sentenceStart);
            dictionary.Add(sentenceEnd, dictionary.Count);
            list.Add(sentenceEnd); 
            foreach (var w in words)
            {
                dictionary.Add(w, dictionary.Count);
                list.Add(w);
            }
        }

        public int GetStartIndex()
        {
            return GetIndex(sentenceStart);
        }

        public int GetEndIndex()
        {
            return GetIndex(sentenceEnd);
        }

        public int GetIndex(string word)
        {
            return dictionary.ContainsKey(word) ? dictionary[word] : dictionary.Count;
        }

        public string GetWord(int index)
        {
            return index < list.Count ? list[index] : unknownToken;
        }
    }
}
