using MathNet.Numerics.LinearAlgebra;
using System;

namespace AnyFileRNN
{
    [Serializable]
    /// serializuje stavy matic 
    class State 
    {
        public Vector<double> h {get; set; } //hd
        public Matrix<double> W { get; set; } //hd, hd
        public Matrix<double> U { get; set; } //hd, id
        public Matrix<double> V { get; set; } //od, hd
        public Vector<double> bh { get; set; } // hd
        public Vector<double> bo { get; set; } // od
    }
}
