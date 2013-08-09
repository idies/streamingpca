using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Jhu.Pca
{
    public class DualPca : PcaBase
    {
        //public int p;
        public int baseSize;

        // Istvan Csabai August 03, 2010
        // We have p of dim dimensional data vectors.
        // If n is large, but p is not large, simple PCA does not work because of the n x n covariance matrix.
        // The dual PCA uses only p x p covariance martix and p x n matrices.
        // baseSize, is the number of returned pca components, it should be equal or less than p
        // A quick summary of Dual PCA can be found at 
        // http://www.math.uwaterloo.ca/~aghodsib/courses/f06stat890/readings/tutorial_stat890.pdf
        // (A copy is checked in with the code, too.)
        // XXX gappy (Mask) might not work correctly
        /*public DualPca(int p, int baseSize)
        {
            //if (baseSize > p) baseSize = p;
            //this.p = p;
            //this.baseSize = baseSize;
        }*/

        public override void Run(IEnumerable<Vector> vectors)
        {
            List<Vector> buffer = new List<Vector>(vectors);

            dim = buffer[0].Value.Length;

            long[] count = new long[dim];
            long[][] ccount = new long[dim][];
            for (int i = 0; i < dim; i++)
                ccount[i] = new long[dim];

            m = new double[dim];
            s2 = new double[dim];
            C = new MathNet.Numerics.LinearAlgebra.Double.DenseMatrix(buffer.Count);

            // Calculate mean vector and covariance matrix
            {
                int k = 0;
                foreach (Vector x in buffer) // runs for 0..(p-1) vectors
                {
                    //for (int i = 0; i < p; i++)
                    //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                    for (int i = 0; i < dim; i ++)
                    {
                        if (x.Mask == null || !x.Mask[i])
                        {
                            count[i]++;
                            m[i] += x.Value[i];
                        }
                    }
                    //);
                }
                foreach (Vector x in buffer) // runs for 0..(p-1) vectors
                {
                    int l = 0;
                    foreach (Vector y in buffer) // runs for 0..(p-1) vectors
                    {
                        //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                        for (int i = 0; i < dim; i ++)
                        {
                            if ((x.Mask == null || !x.Mask[i]) && (y.Mask == null || !y.Mask[i]))
                            {
                                ccount[k][l]++;
                                C[k, l] += (x.Value[i] - m[i]) * (y.Value[i] - m[i]);
                            }
                        }
                        //);
                        l++;
                    }
                    k++;
                }
            }

            for (int i = 0; i < dim; i++)
            {
                m[i] /= (double)count[i];
                s2[i] = C[i, i] / (double)count[i];
            }

            for (int i = 0; i < buffer.Count; i++)
            {
                for (int j = 0; j < buffer.Count; j++)
                {
                    C[i, j] /= (double)ccount[i][j];
                    if (double.IsNaN(C[i, j]))
                    {
                        Console.Write("!");
                        C[i, j] = 1;
                    }
                }
            }

            // ?? in the dual version this cannot be done in one step, it is already done above, maybe can be optimized 
            //if (subtractAverage)
            //{
            //    for (int i = 0; i < dim; i++)
            //    {
            //        for (int j = 0; j < dim; j++)
            //        {
            //            C[i, j] -= m[i] * m[j];
            //            if (double.IsNaN(C[i, j]))
            //            {
            //                Console.Write("!");
            //                C[i, j] = 0;
            //            }
            //        }
            //    }
            //}

            // Do SVD to get eigenbasis and eigenvalues
            MathNet.Numerics.LinearAlgebra.Double.Factorization.DenseSvd svd = new MathNet.Numerics.LinearAlgebra.Double.Factorization.DenseSvd(C, true);

            // eigenvalues are same for the simple and dual PCA, but we return only the top baseSize of them
            L = new MathNet.Numerics.LinearAlgebra.Double.DenseVector(svd.S().SubVector(0, baseSize));

            // the PCA basis can be calculated from the data and the dual eigenbasis
            // we do not need both Vt and U, actually they are just transpose of each other, since C was a real symmetric matrix
            // Vt = svd.Vt;                    
            //E = svd.U;
            E = new MathNet.Numerics.LinearAlgebra.Double.DenseMatrix(dim, buffer.Count);
            for (int k = 0; k < baseSize; k++)
            {
                double eiNorm = 0;
                if (svd.S()[k] > 1e-15)
                {
                    eiNorm = (1.0 / Math.Sqrt(svd.S()[k]));
                }
                //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                for (int i = 0; i < dim; i++)
                {
                    int j = 0;
                    double sum = 0;
                    foreach (Vector x in vectors) // runs for 0..(p-1) vectors
                    {
                        sum += svd.VT()[j, k] * x.Value[i] * eiNorm;
                        j++;
                    }
                    E[i, k] = sum;
                }
                //);
            }

            n = count;
        }

#if false
        public override void Run(Vector[] vectors)
        {
            int[] count = new int[dim];
            int[][] ccount = new int[dim][];
            for (int i = 0; i < dim; i++)
                ccount[i] = new int[dim];

            m = new double[dim];
            C = new MathNet.Numerics.LinearAlgebra.Double.Matrix(p);

            // Calculate mean vector and covariance matrix
            {
                int k = 0;
                foreach (Vector x in vectors) // runs for 0..(p-1) vectors
                {
                    for (int i = 0; i < dim; i++)
                    //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                    {
                        if (x.Mask == null || !x.Mask[i])
                        {
                            count[i]++;
                            m[i] += x.Value[i];
                        }
                    }
                    //);
                }
                foreach (Vector x in vectors) // runs for 0..(p-1) vectors
                {
                    int l = 0;
                    foreach (Vector y in vectors) // runs for 0..(p-1) vectors
                    {
                        for (int i = 0; i < dim; i++)
                        //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                        {
                            if ((x.Mask == null || !x.Mask[i]) && (y.Mask == null || !y.Mask[i]))
                            {
                                ccount[k][l]++;
                                C[k, l] += (x.Value[i] - m[i]) * (y.Value[i] - m[i]);
                            }
                        }
                        //);
                        l++;
                    }
                    k++;
                }
            }
            for (int i = 0; i < dim; i++)
            {
                m[i] /= (double)count[i];
            }

            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    C[i, j] /= (double)ccount[i][j];
                    if (double.IsNaN(C[i, j]))
                    {
                        Console.Write("!");
                        C[i, j] = 1;
                    }
                }
            }

            // ?? in the dual version this cannot be done in one step, it is already done above, maybe can be optimized 
            //if (subtractAverage)
            //{
            //    for (int i = 0; i < dim; i++)
            //    {
            //        for (int j = 0; j < dim; j++)
            //        {
            //            C[i, j] -= m[i] * m[j];
            //            if (double.IsNaN(C[i, j]))
            //            {
            //                Console.Write("!");
            //                C[i, j] = 0;
            //            }
            //        }
            //    }
            //}

            // Do SVD to get eigenbasis and eigenvalues
            Lapack.Svd svd = new Lapack.Svd(C, implementation);

            // eigenvalues are same for the simple and dual PCA, but we return only the top baseSize of them
            double[] d = new double[baseSize];
            for (int i = 0; i < baseSize; i++)
            {
                d[i] = svd.Diagonal[i];
            }
            //L = new MathNet.Numerics.LinearAlgebra.Double.Matrix(svd.Diagonal);
            L = new MathNet.Numerics.LinearAlgebra.Double.Matrix(d);

            // the PCA basis can be calculated from the data and the dual eigenbasis
            // we do not need both Vt and U, actually they are just transpose of each other, since C was a real symmetric matrix
            // Vt = svd.Vt;                    
            //E = svd.U;
            E = new MathNet.Numerics.LinearAlgebra.Double.Matrix(p);
            for (int k = 0; k < baseSize; k++)
            {
                double eiNorm = 0;
                if (svd.Diagonal[k] > 1e-15)
                {
                    eiNorm = (1.0 / Math.Sqrt(svd.Diagonal[k]));
                }
                //new ParallelLib.ParallelFor(0, dim).Execute(delegate(int i)
                for (int i = 0; i < dim; i++)
                {
                    int j = 0;
                    double sum = 0;
                    foreach (Vector x in vectors) // runs for 0..(p-1) vectors
                    {
                        sum += svd.Vt[j, k] * (x.Value[i] - m[i]) * eiNorm;
                        j++;
                    }
                    E[i, k] = sum;
                }
                //);
            }
        }
#endif
    }
}
