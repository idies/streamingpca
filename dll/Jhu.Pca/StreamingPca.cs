using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Double.Factorization;

namespace Jhu.Pca
{
    public class StreamingPca : RobustPca
    {
        private bool basisInizialized;
        private IEnumerator<Vector> vectors;

        public double alpha;

        private double v, s, u;

        public StreamingPca(int p, int q)
            : base(p, q)
        {
        }

        private void InitializeMembers()
        {
            this.basisInizialized = false;
            this.vectors = null;
        }

        public void InitializePca(ParallelQuery<Vector> vectors, int init, InitializationMethod method)
        {
            this.vectors = vectors.WithDegreeOfParallelism(1).GetEnumerator();

            if (!basisInizialized && init > 0)
            {
                // Take first n vectors and initialize
                List<Vector> buffer = new List<Vector>(2 * init);
                int i = 0;
                while (i < init && this.vectors.MoveNext())
                {
                    buffer.Add(this.vectors.Current);
                    i++;
                }

                InitializeBasis(buffer, method);
            }
        }

        private void InitializeBasis(IList<Vector> vectors, InitializationMethod method)
        {
            switch (method)
            {
                case InitializationMethod.DataMatrixSVD:
                    InitWithDataMatrixSVD(vectors);
                    break;
                case InitializationMethod.SimplePCA:
                    InitWithSimplePCA(vectors);
                    break;
                case InitializationMethod.DualPCA:
                    InitWithDualPCA(vectors);
                    break;
                case InitializationMethod.RobustPCA:
                    InitWithRobustPCA(vectors);
                    break;
                default:
                    throw new NotImplementedException();
            }

            this.basisInizialized = true;
        }

        private void InitWithDataMatrixSVD(IList<Vector> buffer)
        {
            // --- compute average
            double[] avg = null;
            double[] var = null;
            long[] count = null;

            int i = 0;
            foreach (Vector vv in buffer)
            {
                if (i == 0)
                {
                    avg = new double[vv.Value.Length];
                    var = new double[vv.Value.Length];
                    count = new long[vv.Value.Length];
                }

                for (int k = 0; k < avg.Length; k++)
                {
                    // Check if point is masked
                    if (vv.Mask == null || !vv.Mask[k])
                    {
                        avg[k] += vv.Value[k];
                        var[k] += vv.Value[k] * vv.Value[k];
                        count[k]++;
                    }
                }

                i++;
            }

            for (int k = 0; k < avg.Length; k++)
            {
                if ((double)count[k] != 0)
                {
                    avg[k] /= (double)count[k];
                    var[k] /= (double)count[k];
                    var[k] -= avg[k] * avg[k];
                }
                else
                {
                    avg[k] = 0;
                    var[k] = 0;
                }
            }

            // Debug code

            using (var f = new System.IO.StreamWriter("avg.txt"))
            {
                for (int j = 0; j < avg.Length; j++)
                {
                    f.WriteLine(avg[j]);
                }
            }

            // End debug code 

            // --- compute median
            // median might be more robust than the average but average is enough
#if false
            List<double>[] flux = null;
            int i = 0;

            while (i < init && this.vectors.MoveNext())
            {
                if (i == 0)
                {
                    flux = new List<double>[this.vectors.Current.Value.Length];
                    for (int k = 0; k < flux.Length; k++)
                    {
                        flux[k] = new List<double>();
                    }
                }

                buffer.Add(this.vectors.Current);

                for (int k = 0; k < flux.Length; k++)
                {
                    Vector vv = this.vectors.Current;

                    // Check if point is masked
                    if (vv.Mask == null || !vv.Mask[k])
                    {
                        flux[k].Add(this.vectors.Current.Value[k]);
                    }
                }

                i++;
            }

            // Compute median
            double[] avg = new double[flux.Length];
            for (int k = 0; k < avg.Length; k++)
            {
                if (flux[k].Count > 0)
                {
                    flux[k].Sort();
                    avg[k] = flux[k][flux[k].Count / 2];
                }
                else
                {
                    avg[k] = 0;
                }
            }
#endif

            // Do SVD on the data matrix to get initial basis
            DenseMatrix M = null;
            i = 0;
            foreach (Vector vv in buffer)
            {
                if (i == 0)
                {
                    M = new DenseMatrix(vv.Value.Length, buffer.Count);
                }

                for (int k = 0; k < avg.Length; k++)
                {
                    // Check if point is masked
                    if (vv.Mask == null || !vv.Mask[k])
                    {
                        if (subtractAverage)
                        {
                            vv.Value[k] -= avg[k];
                        }
                    }
                    else
                    {
                        if (subtractAverage)
                        {
                            // Substitute with average which is 0 now by definition
                            vv.Value[k] = 0;
                        }
                        else
                        {
                            vv.Value[k] = avg[k];
                        }
                    }
                }

                M.SetColumn(i, vv.Value);

                i++;
            }

            DenseSvd svd = new DenseSvd(M, true);

            // Calculate init values

            m = avg;
            s2 = var;
            n = count;
            Vt = (DenseMatrix)svd.VT();
            L = (DenseVector)svd.S();
            E = (DenseMatrix)svd.U();

            dim = avg.Length;

            InitStreamingVariables(buffer);
        }

        private void InitWithSimplePCA(IList<Vector> buffer)
        {
            // Initialize with simple PCA
            SimplePca spca = new SimplePca();
            spca.subtractAverage = this.subtractAverage;
            spca.Run(buffer);

            m = spca.m;
            s2 = spca.s2;
            n = spca.n;
            E = spca.E;
            L = spca.L;

            InitStreamingVariables(buffer);
        }

        private void InitWithRobustPCA(IList<Vector> buffer)
        {
            RobustPca pca = new RobustPca(p, q);
            pca.subtractAverage = this.subtractAverage;
            pca.delta = this.delta;
            pca.W = this.W;
            pca.Wstar = this.Wstar;

            pca.Run(buffer, 2);

            u = buffer.Count;
            v = pca.wsum;
            s = pca.wr2sum;

            m = pca.m;
            s2 = pca.s2;
            n = pca.n;
            E = pca.E;
            L = pca.L;
            sigma2 = pca.sigma2;
            dim = pca.dim;

            step = buffer.Count;
        }

        private void InitWithDualPCA(IList<Vector> buffer)
        {
            DualPca pca = new DualPca();
            pca.subtractAverage = this.subtractAverage;
            pca.Run(buffer);

            m = pca.m;
            s2 = pca.s2;
            n = pca.n;
            E = pca.E;
            L = pca.L;

            dim = pca.dim;

            InitStreamingVariables(buffer);
        }

        private void InitWithSinus()
        {
            // *** needs review
            throw new NotImplementedException();

            E = new MathNet.Numerics.LinearAlgebra.Double.DenseMatrix(base.dim);
            L = new MathNet.Numerics.LinearAlgebra.Double.DenseVector(base.dim);
            for (int i = 0; i < E.ColumnCount; i++)
            {
                L[i] = 1 / (double)(i + 2);

                for (int j = 0; j < E.RowCount; j++)
                {
                    E[j, i] = Math.Sin(Math.PI / (double)base.dim * (double)j * ((double)i + 1.0));
                }
            }


            m = new double[base.dim];
            s2 = new double[base.dim];
            n = new long[base.dim];

            step = 0;
            u = 0;
            v = 0;
            s = 0;
            sigma2 = 1.0;

            // InitStreamingVariables(buffer); *** TODO why?
        }

        private void InitStreamingVariables(IList<Vector> buffer)
        {
            int i = 0;
            
            foreach (Vector vv in buffer)
            {
                if (i == 0)
                {
                    u = v = s = 0;

                    sigma2 = 0;
                    double aa = 0;
                    for (int k = 0; k < vv.Value.Length; k++)
                    {
                        aa += vv.Value[k];
                        sigma2 += vv.Value[k] * vv.Value[k];
                    }
                    aa /= (double)vv.Value.Length;
                    sigma2 /= (double)vv.Value.Length;
                    sigma2 -= aa * aa;
                }

                double[] vp, vpq;
                double r2, w, ws;
                double g1, g2, g3;
                double nv, ns, nu, nsigma2;
                if (!IterateStreamingVariables(vv, out vp, out vpq,
                        out r2, out w, out ws, out g1, out g2, out g3,
                        out nv, out ns, out nu, out nsigma2))
                {
                    continue;
                }

                sigma2 = nsigma2;

                v = nv;
                s = ns;
                u = nu;

                i++;
            }

            step = buffer.Count;
        }

        private bool IterateStreamingVariables(Vector vv, out double[] vp, out double[] vpq,
            out double r2, out double w, out double ws, out double g1, out double g2, out double g3,
            out double nv, out double ns, out double nu, out double nsigma2)
        {

            // Calculate residuals

            vp = ExpandVector(vv, p);
            vpq = ExpandVector(vv, p + q);

            // Debug code
            /*
            using (var f = new System.IO.StreamWriter("vec.txt"))
            {
                for (int j = 0; j < vv.Value.Length; j++)
                {
                    f.WriteLine("{0} {1} {2} {3}", m[j], vv.Value[j], vp[j], vpq[j]);
                }
            }
            */
            // End debug code

            r2 = GetResidual(vv, vp, vpq);

            if (r2 == 0 || double.IsNaN(r2))
            {
                // Seems to be a fully masked spectrum
                w = ws = g1 = g2 = g3 = nv = nu = ns = nsigma2 = 0;
                return false;
            }

            w = W(r2 / sigma2);
            ws = Wstar(r2 / sigma2);

            nv = alpha * v + w;
            ns = alpha * s + w * r2;
            nu = alpha * u + 1;

            g1 = alpha * v / nv;
            g2 = alpha * s / ns;
            g3 = alpha * u / nu;

            nsigma2 = g3 * sigma2 + (1 - g3) * ws * r2 / delta;

            return true;
        }

        public new bool Step()
        {
            if (vectors.MoveNext())
            {
                double[] vp, vpq;
                double r2, w, ws;
                double g1, g2, g3;
                double nv, ns, nu, nsigma2;

                if (!IterateStreamingVariables(vectors.Current, out vp, out vpq,
                        out r2, out w, out ws, out g1, out g2, out g3,
                        out nv, out ns, out nu, out nsigma2))
                {
                    return true;
                }

                // Fill gaps
                if (vectors.Current.Mask != null)
                {
                    for (int i = 0; i < vectors.Current.Value.Length; i++)
                    {
                        if (!vectors.Current.Mask[i])
                        {
                            n[i]++;
                        }
                        else
                        {
                            vectors.Current.Value[i] = vp[i];
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < vectors.Current.Value.Length; i++)
                    {
                        n[i]++;
                    }
                }

                double[] x = vectors.Current.Value;
                double[] y = new double[dim];       // x - mean

                for (int i = 0; i < dim; i++)
                {
                    if (subtractAverage)
                    {
                        y[i] = x[i] - m[i];
                    }
                    else
                    {
                        y[i] = x[i];
                    }
                }

                // Update mean and variance
                double[] nm = new double[dim];
                double[] ns2 = new double[dim];
                for (int i = 0; i < dim; i++)
                {
                    nm[i] = g1 * m[i] + (1 - g1) * x[i];
                    ns2[i] = g1 * s2[i] + (1 - g1) * (x[i] - m[i]) * (x[i] - m[i]);
                }

                // Update eigensystem by doing PCA on a thin matrix
                DenseMatrix A = new DenseMatrix(dim, p + q + 1);
                for (int k = 0; k < p + q; k++)
                {
                    for (int i = 0; i < dim; i++)
                    {
                        A[i, k] = E[i, k] * Math.Sqrt(g2 * L[k]);
                        if (double.IsNaN(A[i, k]))
                        {
                            return false;
                        }
                    }
                }
                for (int i = 0; i < dim; i++)
                {
                    A[i, p + q] = y[i] * Math.Sqrt((1 - g2) * sigma2 / r2);
                    if (double.IsNaN(A[i, p + q]))
                    {
                        return false;
                    }
                }

                DenseSvd svd = new DenseSvd(A, true);

                // Update sigma
                // *** why removed?
                // double nsigma2 = g3 * sigma2 + (1 - g3) * ws * r2 / delta;

                // --- Propagate updates to next iteration ---
                m = nm;
                s2 = ns2;

                L = new MathNet.Numerics.LinearAlgebra.Double.DenseVector(svd.S().Count);
                for (int i = 0; i < svd.S().Count; i++)
                {
                    L[i] = svd.S()[i] * svd.S()[i];
                }

                E = (DenseMatrix)svd.U();

                sigma2 = nsigma2;

                v = nv;
                s = ns;
                u = nu;

                // *** debug code to dump eigenvalues
                /*System.IO.File.AppendAllText("eigs.txt",
                    String.Format("{0}\t{1}\t{2}\t{3}\t{4}\r\n", sigma2, L[0], L[1], L[2], L[3]));*/

                step++;

                return true;
            }
            else
            {
                return false;
            }
        }

        public new void Run(ParallelQuery<Vector> vectors, int init)
        {
            InitializePca(vectors, init, InitializationMethod.DualPCA);
            Run();
        }

        public void Run()
        {
            while (Step()) ;
        }
    }
}
