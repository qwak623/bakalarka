using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnyFileRNN
{
    class Tokenizer : IEnumerable<string>
    {
        string text;
        public Tokenizer(string text)
        {
            this.text = text;
        }

        public IEnumerator<string> GetEnumerator()
        {
            if (text.Length != 0)
            {
                char ch = text[0];
                if (char.IsWhiteSpace(ch))
                    ch = ' ';
                StringBuilder sb = new StringBuilder(ch.ToString());

                for (int i = 1; i < text.Length; i++)
                {
                    if (char.IsLetterOrDigit(ch) && char.IsLetterOrDigit(text[i]))
                    {
                        ch = text[i];
                        sb.Append(ch);
                    }
                    else if (char.IsWhiteSpace(ch) && char.IsWhiteSpace(text[i]))
                    {
                        ch = ' ';
                        sb.Append(ch);
                    }
                    else
                    {
                        yield return sb.ToString();
                        sb.Clear();
                        ch = text[i];
                        if (char.IsWhiteSpace(ch))
                            ch = ' ';
                        sb.Append(ch);
                    }
                }
                
                yield return sb.ToString();
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
