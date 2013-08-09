using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Jhu.Pca
{
    public class SimplePca : PcaBase
    {
        public SimplePca()
            :base()
        {
        }

        public override void Run(IEnumerable<Vector> vectors)
        {
            long[] count = null;
            long[][] ccount = null;

            // Calculate mean vector and covariance matrix
            foreach (Vector x in vectors)
            {
                if (dim == -1)
                {
                    dim = x.Value.Length;

                    count = new long[dim];
                    ccount = new long[dim][];
                    for (int i = 0; i < dim; i++)
                        ccount[i] = new long[dim];

                    m = new double[dim];
                    s2 = new double[dim];
                    C = new DenseMatrix(dim);
                }

                Parallel.For(0, dim, delegate(int i)
                {
                    if (x.Mask == null || !x.Mask[i])
                    {
                        count[i]++;
                        m[i] += x.Value[i];
                    }

                    if (x.Mask == null || !x.Mask[i])
                    {

                        for (int j = 0; j < dim; j++)
                        {
                            if (x.Mask == null || !x.Mask[j])
                            {
                                ccount[i][j]++;
                                C[i, j] += x.Value[i] * x.Value[j];
                            }
                        }
                    }
                });
            }

            for (int i = 0; i < dim; i++)
            {
                // mean
                m[i] /= (double)count[i];

                // variance
                s2[i] = C[i, i] / (double)ccount[i][i] - m[i] * m[i];

                // covariance
                for (int j = 0; j < dim; j++)
                {
                    C[i, j] /= (double)ccount[i][j];
                    if (double.IsNaN(C[i, j]))
                    {
                        Console.Write("!");
                        C[i, j] = 0;
                    }
                }
            }

            if (subtractAverage)
            {
                for (int i = 0; i < dim; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        C[i, j] -= m[i] * m[j];
                        if (double.IsNaN(C[i, j]))
                        {
                            Console.Write("!");
                            C[i, j] = 0;
                        }
                    }
                }
            }

            // Do SVD to get eigenbasis and eigenvalues
            var svd = new MathNet.Numerics.LinearAlgebra.Double.Factorization.DenseSvd(C, true);

            n = count;

            Vt = (DenseMatrix)svd.VT();
            L = (DenseVector)svd.S();
            E = (DenseMatrix)svd.U();
        }
    }
}
