using System.Collections.Generic;

namespace AnyFileRNN
{
    public interface IDataHolder
    {
        List<TrainSequence> epoch { get; set; }
    }

    public class TrainSequence
    {
        public List<int> x = new List<int>();
        public List<int> y = new List<int>();
    }
}
