using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace AnyFileRNN
{
    public static class Functions
    {
        public static Vector<double> ToVector(this int index, int id)
        {
            var vector = Vector<double>.Build.Dense(id);
            vector[index] = 1;
            return vector;
        }

        public static Vector<double> Softmax(Vector<double> v)
        {
            var expList = v.Select(a => Exp(a)).ToList();
            var sumExp = expList.Sum();
            for (int i = 0; i < v.Count; i++)
            {
                v[i] = expList[i] / sumExp;
            }
            return v;
        }

        public static Vector<double> Clip(Vector<double> x)
        {
            //    return x.Map(a => Abs(a) > 5 ? Sign(a) * 5 : a ); tohle je neefektivni, vytvari novy vektor
            for (int i = 0; i < x.Count; i++)
                x[i] = Abs(x[i]) > 5 ? Sign(x[i]) * 5 : x[i];
            return x;
        }

        public static Matrix<double> Clip(Matrix<double> A)
        {
            for (int i = 0; i < A.ColumnCount; i++)
            {
                for (int j = 0; j < A.RowCount; j++)
                {
                    A[j, i] = Abs(A[j, i]) > 5 ? Sign(A[j, i]) * 5 : A[j, i];
                }
            }
            return A;
        }

        public static bool hasNan(this Matrix<double> A)
        {
            for (int i = 0; i < A.ColumnCount; i++)
            {
                if (A.Column(i).Any(a => double.IsNaN(a)))
                    return true;
            }
            return false;
        }

    }
}
