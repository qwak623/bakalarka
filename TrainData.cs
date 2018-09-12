using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;

namespace AnyFileRNN
{
    class TrainData
    {
        public List<TrainSequence> epoch = new List<TrainSequence>();

        public TrainData(string path, Vocabulary vocab)
        {
            using (var reader = new StreamReader(path))
            {
                for (int i = 0; i < 100; i++)
                //while (!reader.EndOfStream)
                {
                    var xTrain = new List<Vector<double>>();
                    var yTrain = new List<Vector<double>>();

                    // pridani sentence start do vektoru x
                    {
                        var x1 = Vector<double>.Build.Dense(vocab.Size);
                        x1[vocab.GetStartIndex()] = 1;
                        xTrain.Add(x1);
                    }

                    var tokenizer = new Tokenizer(reader.ReadLine());
                    foreach (var token in tokenizer.Take(10)) // TODO tady neco s tou delkou
                    {
                        var xyk = Vector<double>.Build.Dense(vocab.Size);
                        xyk[vocab.GetIndex(token)] = 1;
                        xTrain.Add(xyk);
                        yTrain.Add(xyk);
                    }

                    // pridani sentence end do vektoru y
                    {
                        var yn = Vector<double>.Build.Dense(vocab.Size);
                        yn[vocab.GetEndIndex()] = 1;
                        yTrain.Add(yn);
                    }

                    epoch.Add(new TrainSequence { x = xTrain, y = yTrain });
                }
            }
        }

        public class TrainSequence
        {
            public List<Vector<double>> x;
            public List<Vector<double>> y;
        }
    }

}
