using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Jhu.Pca
{
    public class PcaException : Exception
    {
        public PcaException()
            : base()
        {
        }

        public PcaException(string message)
            : base(message)
        {
        }
    }
}
