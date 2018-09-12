using AnyFileRNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using BrightWire.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Runtime.Serialization.Formatters.Binary;
using System.Diagnostics;
using static System.Math;
using static AnyFileRNN.Functions;

namespace AnyFileRNN
{
    public class RNN
    {
        Vector<double> h; //hd
        Matrix<double> U; //hd, id
        Matrix<double> V; //od, hd
        Matrix<double> W; //hd, hd
        Vector<double> bh; // hd
        Vector<double> bo; // od
        readonly int id;
        readonly int od;
        readonly int hd;
        double learningRate;
        readonly double initialLearningRate;

        // ulozny prostor
        List<Vector<double>> list_h = new List<Vector<double>>();
        List<Vector<double>> list_o = new List<Vector<double>>();
        Vocabulary vocab;

        public RNN(Vocabulary vocab, int id, int od, int hd, double learningRate = 0.005)
        {
            this.vocab = vocab;
            this.id = id;
            this.od = od;
            this.hd = hd;
            this.learningRate = learningRate;
            initialLearningRate = learningRate;
            IContinuousDistribution scd = new ContinuousUniform(-1 / Sqrt(id), 1 / Sqrt(id));
            IContinuousDistribution wcd = new ContinuousUniform(-1 / Sqrt(hd), 1 / Sqrt(hd));

            h = Vector<double>.Build.Dense(hd);
            U = Matrix<double>.Build.Random(hd, id, scd);
            V = Matrix<double>.Build.Random(od, hd, wcd);
            W = Matrix<double>.Build.Random(hd, hd, wcd);
            bo = Vector<double>.Build.Dense(od);
            bh = Vector<double>.Build.Dense(hd);
        }

        public void Load(TextReader reader)
        {
            throw new NotImplementedException();
        }

        public List<Vector<double>> ForwProp(List<int> x)
        {
            list_h.Clear();  // TODO tohle je mozna jen zdrzovani
            list_h.Add(Vector<double>.Build.Dense(hd));
            list_o.Clear();
            return x.Select(a => forwPropStep(a)).ToList();
        }

        private Vector<double> forwPropStep(int x)  // x je jen index a ne vektor
        {
            // h_t = tanh(U * x_t + W * h_t-1)
            h = (U * x.ToVector(id) + W * h + bh).Map(a => Tanh(a));
            list_h.Add(h);

            var o = Softmax(V * h + bo);
            list_o.Add(o);
            return o;
        }

        public void BackPropTT(List<int> x, List<int> y, List<Vector<double>> o)
        {
            var dU = Matrix<double>.Build.Dense(hd, id);
            var dV = Matrix<double>.Build.Dense(od, hd);
            var dW = Matrix<double>.Build.Dense(hd, hd);
            var dbh = Vector<double>.Build.Dense(hd);
            var dbo = Vector<double>.Build.Dense(od);
            var dhNext = Vector<double>.Build.Dense(hd);

            for (int t = o.Count - 1; t >= 0; t--)
            {
                Vector<double> dy = Vector<double>.Build.Dense(od);
                list_o[t].CopyTo(dy);
                //dy -= y[t]; // vektorova verze
                dy[y[t]] -= 1; // indexova verze

                dV += dy.OuterProduct(list_h[t + 1]);
                dbo += dy;
                var dh = dy * V + dhNext;

                //var dhraw = (1 - list_h[t + 1] * list_h[t + 1]) * dh;
                var dhraw = Vector<double>.Build.Dense(hd);
                var neco = list_h[t + 1].Map(a => 1 - a * a);
                neco.Map2((a, b) => a * b, dh, dhraw);

                dbh += dhraw;
                dU += dhraw.OuterProduct(x[t].ToVector(id));
                dW += dhraw.OuterProduct(list_h[t]);
                dhNext = W * dhraw;
            }


            //if (dU.ForAll(a => Abs(a) > 5))
            //    U = U;
            //if (dV.ForAll(a => Abs(a) > 5))
            //    U = U;
            //if (dW.ForAll(a => Abs(a) > 5))
            //    U = U;
            //if (dbh.ForAll(a => Abs(a) > 5))
            //    U = U;
            //if (dbo.ForAll(a => Abs(a) > 5))
            //    U = U;

            //U = U - learningRate * dU;
            //V = V - learningRate * dV;
            //W = W - learningRate * dW;
            //bh = bh - learningRate * dbh;
            //bo = bo - learningRate * dbo;

            U = Clip(U - learningRate * dU.Map(a => a / Sqrt(a * a + 1e-8)));
            V = Clip(V - learningRate * dV.Map(a => a / Sqrt(a * a + 1e-8)));
            W = Clip(W - learningRate * dW.Map(a => a / Sqrt(a * a + 1e-8)));
            bh = Clip(bh - learningRate * dbh.Map(a => a / Sqrt(a * a + 1e-8)));
            bo = Clip(bo - learningRate * dbo.Map(a => a / Sqrt(a * a + 1e-8)));
        }

        public void UpdateLearningRate()
        {
            learningRate = learningRate < 2e-10 ? initialLearningRate : learningRate / 2;
        }

        public void Save(string path)
        {
            try
            {
                using (var fs = new FileStream(path, FileMode.Create))
                {
                    BinaryFormatter bf = new BinaryFormatter();
                    bf.Serialize(fs, new State
                    {
                        h = h,
                        U = U,
                        V = V,
                        W = W,
                        bo = bo,
                        bh = bh
                    });
                }
                Console.WriteLine($"Saving to file '{path}' was succesful.");
            }
            catch
            {
                Console.WriteLine($"Saving to file '{path}' failed.");
            }
        }

        public void Load(string path)
        {
            try
            {
                using (var fs = new FileStream(path, FileMode.Open))
                {
                    BinaryFormatter bf = new BinaryFormatter();
                    State state = (State)bf.Deserialize(fs);
                    h = state.h;
                    U = state.U;
                    V = state.V;
                    W = state.W;
                    bo = state.bo;
                    bh = state.bh;
                }
                Console.WriteLine($"Loading from file '{path}' was succesful.");
            }
            catch
            {
                Console.WriteLine($"Loading from file '{path}' failed.");
                IContinuousDistribution scd = new ContinuousUniform(-1 / Sqrt(id), 1 / Sqrt(id));
                IContinuousDistribution wcd = new ContinuousUniform(-1 / Sqrt(hd), 1 / Sqrt(hd));
                h = Vector<double>.Build.Dense(hd);
                U = Matrix<double>.Build.Random(hd, id, scd);
                V = Matrix<double>.Build.Random(od, hd, wcd);
                W = Matrix<double>.Build.Random(hd, hd, wcd);
                bo = Vector<double>.Build.Dense(hd);
                bh = Vector<double>.Build.Dense(od);
            }
        }

        public double CalculateLoss(List<Vector<double>> o, List<int> y)  // cross entropy loss
        {
            return o.Zip(y, (a, b) => -Log(a[b])).Sum() / y.Count;
        }

        public void TrainEpochWithLossWriting(int epochCount, IDataHolder data)
        {
            int epochNumber = 0;

            var watch = new Stopwatch();

            double totalEpochTime = 0;
            double avgEpochTime;

            while (epochNumber < epochCount)
            {
                watch.Restart();
                double totalLoss = 0;
                foreach (var seq in data.epoch)
                {
                    // forward prop
                    var o = ForwProp(seq.x);
                    totalLoss += CalculateLoss(o, seq.y);

                    if (o.Any(a => vocab.GetWord(a.MaximumIndex()) == "_UNKNOWN_"))
                        o = o;

                    // backprop
                    BackPropTT(seq.x, seq.y, o);
                }

                watch.Stop();
                double epochTime = watch.ElapsedMilliseconds;
                totalEpochTime += epochTime;

                if (epochNumber % 1 == 0)
                    Console.WriteLine($"epoch: {epochNumber}, loss: {totalLoss}, timeElapsed: {epochTime}ms");

                epochNumber++;
            }

            avgEpochTime = totalEpochTime / epochCount;
            Console.WriteLine($"average epoch time: {avgEpochTime}ms");
        }

        public void TrainEpoch(int epochCount, IDataHolder data)
        {
            int epochNumber = 0;
            while (epochNumber < epochCount)
            {
                foreach (var seq in data.epoch)
                {
                    var o = ForwProp(seq.x);
                    foreach (var item in seq.x)
                        Console.Write(vocab.GetWord(item));
                    Console.WriteLine();

                    BackPropTT(seq.x, seq.y, o);
                }
                epochNumber++;
                UpdateLearningRate();
            }
        }

        public void TrainEpochWithOutput(TextWriter writer, IDataHolder data)
        {
            foreach (var seq in data.epoch)
            {
                // forward prop
                var o = ForwProp(seq.x);

                // print output
                foreach (var item in o)
                    writer.Write(vocab.GetWord(item.MaximumIndex()));
                writer.WriteLine();

                // backprop
                BackPropTT(seq.x, seq.y, o);
            }
            Console.WriteLine($"output written.");
            writer.Flush();
        }
    }
}