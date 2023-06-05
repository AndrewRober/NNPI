namespace NNPI.Kernel.NNPIMath.Linear_Algebra
{
    public class EigenvalueDecomposition
    {
        public enum DecompositionMethod
        {
            QR,
            Jacobi
        }

        public Vector EigenValues { get; private set; }
        public Matrix EigenVectors { get; private set; }

        /// <summary>
        /// Initializes a new instance of the EigenvalueDecomposition class.
        /// </summary>
        /// <param name="matrix">The square matrix to decompose.</param>
        /// <param name="method">The decomposition method to use. Defaults to DecompositionMethod.QR.</param>
        public EigenvalueDecomposition(Matrix matrix, DecompositionMethod method = DecompositionMethod.QR)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (matrix.RowCount != matrix.ColumnCount)
            {
                throw new ArgumentException("The input matrix must be square.", nameof(matrix));
            }

            switch (method)
            {
                case DecompositionMethod.QR:
                    ComputeEigenvaluesQR(matrix);
                    break;
                case DecompositionMethod.Jacobi:
                    ComputeEigenvaluesJacobi(matrix);
                    break;
                default:
                    throw new ArgumentException("Invalid method. Use DecompositionMethod.QR or DecompositionMethod.Jacobi.", nameof(method));
            }
        }

        /// <summary>
        /// Computes the eigenvalues and eigenvectors of a square matrix using the QR algorithm.
        /// </summary>
        /// <param name="A">The square matrix to decompose.</param>
        private void ComputeEigenvaluesQR(Matrix A)
        {
            int n = A.RowCount;
            Matrix Ak = A.Copy();
            EigenVectors = Matrix.Identity(n);

            // Adjust the number of iterations for convergence as needed
            for (int k = 0; k < 100; k++)
            {
                var (Q, R) = QRDecomposition(Ak);
                Ak = R * Q;
                EigenVectors *= Q;
            }

            EigenValues = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                EigenValues[i] = Ak[i, i];
            }
        }

        /// <summary>
        /// Computes the QR decomposition of a square matrix.
        /// </summary>
        /// <param name="A">The square matrix to decompose.</param>
        /// <returns>A tuple containing the orthogonal matrix Q and the upper triangular matrix R.</returns>
        private static (Matrix Q, Matrix R) QRDecomposition(Matrix A)
        {
            if (A == null)
            {
                throw new ArgumentNullException(nameof(A));
            }

            if (A.RowCount != A.ColumnCount)
            {
                throw new ArgumentException("The input matrix must be square.", nameof(A));
            }

            int n = A.RowCount;
            Matrix Q = Matrix.Identity(n);
            Matrix R = A.Copy();

            for (int k = 0; k < n - 1; k++)
            {
                Vector x = R.GetColumn(k).SubVector(k);
                Vector e = Vector.Basis(n - k, 0);
                Vector v = x + x[0] < 0 ? -x.Norm() * e : x.Norm() * e;
                Matrix F = Matrix.Identity(n - k) - 2 * (v.OuterProduct(v).DivideByScalar(v.DotProduct(v)));
                Matrix Qk = Matrix.Identity(n).SetSubMatrix(k, k, F);
                Q = Q * Qk;
                R = Qk * R;
            }

            return (Q, R);
        }

        /// <summary>
        /// Computes the eigenvalues and eigenvectors of a square matrix using the Jacobi method.
        /// </summary>
        /// <param name="A">The square matrix to decompose.</param>
        private void ComputeEigenvaluesJacobi(Matrix A)
        {
            int n = A.RowCount;
            Matrix Ak = A.Copy();
            EigenVectors = Matrix.Identity(n);

            // Adjust the number of iterations for convergence as needed
            for (int k = 0; k < 100; k++)
            {
                int p, q;
                FindLargestOffDiagonalElement(Ak, out p, out q);

                if (Ak[p, q] == 0) break;

                double phi = 0.5 * Math.Atan2(2 * Ak[p, q], Ak[q, q] - Ak[p, p]);
                Matrix J = Matrix.Identity(n);
                J[p, p] = J[q, q] = Math.Cos(phi);
                J[p, q] = Math.Sin(phi);
                J[q, p] = -Math.Sin(phi);

                Ak = J.Transpose() * Ak * J;
                EigenVectors *= J;
            }

            EigenValues = new Vector(n);
            for (int i = 0; i < n; i++)
            {
                EigenValues[i] = Ak[i, i];
            }
        }

        /// <summary>
        /// Finds the row and column indices of the largest off-diagonal element in a square matrix.
        /// </summary>
        /// <param name="A">The square matrix to search.</param>
        /// <param name="row">The row index of the largest off-diagonal element.</param>
        /// <param name="col">The column index of the largest off-diagonal element.</param>
        private void FindLargestOffDiagonalElement(Matrix A, out int row, out int col)
        {
            int n = A.RowCount;
            row = col = 0;
            double maxValue = 0;

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Math.Abs(A[i, j]) > maxValue)
                    {
                        maxValue = Math.Abs(A[i, j]);
                        row = i;
                        col = j;
                    }
                }
            }
        }
    }
}
