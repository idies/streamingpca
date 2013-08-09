using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Double.Factorization;

namespace Jhu.Pca
{
    public class RobustPca : PcaBase
    {
        public delegate double WeightFunction(double t);

        private List<Vector> vectors;

        public double sigma2;
        public double delta;
        public WeightFunction W;
        public WeightFunction Wstar;
        public int p, q;

        public double wsum = 0;
        public double wr2sum = 0;

        protected int step;

        public RobustPca(int p, int q)
        {
            this.p = p;
            this.q = q;
        }

        public void Init(IEnumerable<Vector> vectors)
        {
            this.vectors = new List<Vector>(vectors);

            // Initialize with classic PCA
            SimplePca pca = new SimplePca();
            pca.Run(this.vectors);

            // Copy results of classic PCA
            this.m = pca.m;
            this.s2 = pca.s2;
            this.n = pca.n;
            this.L = pca.L;
            this.E = pca.E;
            this.dim = pca.dim;

            // Initialize sigma2 from classic eigenvalue residuals
            sigma2 = 0;
            for (int i = p; i < dim; i++)
            {
                sigma2 += L[i];
            }

            Console.WriteLine(sigma2);
        }

        public bool Step()
        {
            double[] nm = new double[dim];
            double[] ns2 = new double[dim];
            DenseMatrix nC = new DenseMatrix(dim);
            double[] r2 = new double[vectors.Count];
            double[] w = new double[vectors.Count];

            // Calculate residuals
            double[] vp, vpq;
            for (int k = 0; k < vectors.Count; k++)
            {
                vp = ExpandVector(vectors[k], p);
                vpq = ExpandVector(vectors[k], p + q);

                r2[k] = GetResidual(vectors[k], vp, vpq);

                if (double.IsNaN(r2[k]) || r2[k] == 0)
                {
                    r2[k] = 100;
                }

                // fill gaps
                for (int i = 0; i < vectors[k].Value.Length; i++)
                {
                    if (vectors[k].Mask == null || !vectors[k].Mask[i])
                    {
                        n[k]++;
                    }
                    else
                    {
                        vectors[k].Value[i] = vp[i];
                    }
                }
            }

            // Calculate new sigma2
            for (int i = 0; i < 10; i++)
            {
                double nsigma2 = 0;
                for (int n = 0; n < vectors.Count; n++)
                {
                    double ws = Wstar(r2[n] / sigma2);
                    nsigma2 += ws * r2[n];
                }

                sigma2 = nsigma2 / vectors.Count / delta;
            }

            // Calculate weights and average
            wsum = 0;
            wr2sum = 0;
            for (int n = 0; n < vectors.Count; n++)
            {
                double[] x = vectors[n].Value;

                w[n] = W(r2[n] / sigma2);

                for (int i = 0; i < dim; i++)
                {
                    nm[i] += w[n] * x[i];
                    ns2[i] += w[n] * (x[i] - nm[i]) * (x[i] - nm[i]);
                }

                wsum += w[n];
                wr2sum += w[n] * r2[n];
            }
            for (int i = 0; i < dim; i++)
            {
                nm[i] /= wsum;
                ns2[i] /= wsum;
            }

            // Calculate covariance matrix

            for (int n = 0; n < vectors.Count; n++)
            {
                double[] x = vectors[n].Value;

                for (int i = 0; i < dim; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        if (subtractAverage)
                        {
                            nC[i, j] += sigma2 / wr2sum * (w[n] * (x[i] - nm[i]) * (x[j] - nm[j]));
                        }
                        else
                        {
                            nC[i, j] += sigma2 / wr2sum * (w[n] * x[i] * x[j]);
                        }
                    }
                }
            }


            // Do SVD on covariance matrix
            DenseSvd svd = new DenseSvd(nC, true);
            this.m = nm;
            this.s2 = ns2;
            this.E = (DenseMatrix)svd.U();
            this.L = (DenseVector)svd.S();
            this.Vt = (DenseMatrix)svd.VT();
            this.C = nC;

            return true;
        }

        public override void Run(IEnumerable<Vector> vectors)
        {
            Run(vectors, 3);   //***
        }

        public void Run(IEnumerable<Vector> vectors, int steps)
        {
            Init(vectors);
            for (int i = 0; i < steps; i++)
            {
                Step();
            }
        }

        protected MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> GetProjector()
        {
            MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> Ep = E.SubMatrix(0, dim - 1, 0, p - 1);
            MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> P = Ep * Ep.Transpose();
            return P;
        }

        protected double GetResidual(Vector vector, double[] vp, double[] vpq)
        {
            double res = 0;

            if (vector.Mask == null)
            {
                for (int i = 0; i < vector.Value.Length; i++)
                {
                    n[i]++;
                    double r = vector.Value[i] - vp[i];
                    res += r * r;
                }
            }
            else
            {
                for (int i = 0; i < vector.Value.Length; i++)
                {
                    double r;
                    if (vector.Mask[i])
                    {
                        r = vp[i] - vpq[i];
                    }
                    else
                    {
                        r = vector.Value[i] - vp[i];
                    }

                    res += r * r;
                }
            }

            // TODO: Normalize with norm^2?

            return res;
        }

        protected double[] ExpandVector(Vector v, int pp)
        {
            MathNet.Numerics.LinearAlgebra.Double.DenseMatrix M = new MathNet.Numerics.LinearAlgebra.Double.DenseMatrix(pp);
            MathNet.Numerics.LinearAlgebra.Double.DenseVector F = new MathNet.Numerics.LinearAlgebra.Double.DenseVector(pp);

            // Calculate weight vector
            double[] w = new double[v.Value.Length];
            for (int l = 0; l < v.Value.Length; l++)
            {
                w[l] = v.Mask == null || v.Mask[l] ? 0.0 : v.Weight == null ? 1.0 : v.Weight[l];
            }

            for (int i = 0; i < M.RowCount; i++)
            {
                double[] ei = E.Column(i).ToArray();

                for (int j = 0; j < M.ColumnCount; j++)
                {
                    double[] ej = E.Column(j).ToArray();

                    double m = 0;

                    for (int l = 0; l < v.Value.Length; l++)
                    {
                        m += w[l] * ei[l] * ej[l];
                    }

                    M[i, j] = m;
                }

                double f = 0;
                for (int l = 0; l < v.Value.Length; l++)
                {
                    if (subtractAverage)
                    {
                        f += w[l] * (v.Value[l] - m[l]) * ei[l];
                    }
                    else
                    {
                        f += w[l] * v.Value[l] * ei[l];
                    }
                }
                F[i] = f;
            }

            DenseSvd svd = new DenseSvd(M, true);
            DenseVector x = (DenseVector)svd.Solve(F);

            // This multiplication is done on the truncated basis!
            //MathNet.Numerics.LinearAlgebra.Double.Matrix vv = E * a;
            double[] vv = new double[v.Value.Length];
            for (int i = 0; i < vv.Length; i++)
            {
                for (int j = 0; j < pp; j++)
                {
                    vv[i] += E[i, j] * x[j];
                }
            }

            if (subtractAverage)
            {
                for (int l = 0; l < v.Value.Length; l++)
                {
                    vv[l] += m[l];
                }
            }

            // --- debug code
            /*
            using (System.IO.StreamWriter gap = new System.IO.StreamWriter("gapfill.txt"))
            {
                for (int i = 0; i < v.Value.Length; i++)
                {
                    gap.WriteLine("{0}\t{1}\t{2}", i, v.Value[i], vv[i]);
                }
            }

            using (System.IO.StreamWriter eig = new System.IO.StreamWriter("eig.txt"))
            {
                for (int i = 0; i < E.Rows; i++)
                {
                    eig.Write("{0}", i);
                    for (int j = 0; j < E.Columns; j++)
                    {
                        eig.Write("\t{0}", E[i, j]);
                    }
                    eig.WriteLine();
                }
            }
             * */
            // --- debug code end

            return vv;

        }

    }
}

