using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Jhu.Pca
{
    public abstract class PcaBase
    {
        public int dim;
        public bool subtractAverage;
       
        public double[] m;
        public double[] s2;
        public long[] n;
        public MathNet.Numerics.LinearAlgebra.Double.DenseMatrix C;
        public MathNet.Numerics.LinearAlgebra.Double.DenseMatrix Vt;
        public MathNet.Numerics.LinearAlgebra.Double.DenseVector L;
        public MathNet.Numerics.LinearAlgebra.Double.DenseMatrix E;

        public PcaBase()
        {
            InitializeMembers();
        }

        public PcaBase(int dim)
        {
            InitializeMembers();

            this.dim = dim;
        }

        private void InitializeMembers()
        {
            dim = -1;
            this.subtractAverage = false;

            this.m = null;
            this.s2 = null;
            this.n = null;
            this.C = null;
            this.Vt = null;
            this.L = null;
            this.E = null;
        }

        public abstract void Run(IEnumerable<Vector> vectors);
    }
}
