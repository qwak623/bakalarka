using System.Collections.Generic;
using System.IO;

namespace AnyFileRNN
{
    class DataHolder : IDataHolder
    {
        public List<TrainSequence> epoch { get; set; } = new List<TrainSequence>();

        public DataHolder(string path, Vocabulary vocab, int seqLen = 10)
        {
            using (var reader = new StreamReader(path))
            {
                for (int i = 0; i < 100; i++)
                //while (!reader.EndOfStream)
                {
                    var tokenizer = new Tokenizer(reader.ReadLine());
                    var seq = new TrainSequence();

                    foreach (var token in tokenizer)
                    {
                        if (seq.x.Count == 0)
                            seq.x.Add(vocab.GetStartIndex());
                        if (vocab.GetIndex(token) == vocab.Size - 1 || token == " ") // TODO aby tu nebyly unknown
                            continue;
                        seq.x.Add(vocab.GetIndex(token));
                        seq.y.Add(vocab.GetIndex(token));
                        if (seq.x.Count == seqLen)
                        {
                            seq.y.Add(vocab.GetEndIndex());
                            epoch.Add(seq);
                            break;  // TODO
                            seq = new TrainSequence(); // neberu konce vet
                        }
                    }
                    if (seq.x.Count != seqLen)
                    {
                        seq.y.Add(vocab.GetEndIndex());
                        epoch.Add(seq);
                    }
                }
            }
        }
    }

}
