using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Math = System.Math;

namespace NNPI.Kernel.Math.Linear_Algebra
{
    /// <summary>
    /// Represents a mathematical vector and provides basic operations.
    /// </summary>
    public class Vector
    {
        public double[] Data { get; }
        public int Length => Data.Length;

        /// <summary>
        /// Initializes a new instance of the Vector class with the specified length.
        /// </summary>
        /// <param name="length">The length of the vector. Must be positive.</param>
        public Vector(int length)
        {
            if (length <= 0)
            {
                throw new ArgumentException("Length must be positive.", nameof(length));
            }

            Data = new double[length];
        }

        /// <summary>
        /// Initializes a new instance of the Vector class with the specified data.
        /// </summary>
        /// <param name="data">The data of the vector.</param>
        public Vector(double[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
        }

        public double this[int index]
        {
            get => Data[index];
            set => Data[index] = value;
        }
    }

    /// <summary>
    /// Represents a mathematical matrix and provides basic operations.
    /// </summary>
    public class Matrix
    {
        public double[][] Data { get; }
        public int Rows => Data.Length;
        public int Columns => Data[0].Length;

        /// <summary>
        /// Initializes a new instance of the Matrix class with the specified number of rows and columns.
        /// </summary>
        /// <param name="rows">The number of rows in the matrix. Must be positive.</param>
        /// <param name="columns">The number of columns in the matrix. Must be positive.</param>
        public Matrix(int rows, int columns)
        {
            if (rows <= 0)
                throw new ArgumentException("The number of rows must be positive.", nameof(rows));

            if (columns <= 0)
                throw new ArgumentException("The number of columns must be positive.", nameof(columns));

            Data = new double[rows][];
            for (int i = 0; i < rows; i++)
                Data[i] = new double[columns];
        }

        /// <summary>
        /// Initializes a new instance of the Matrix class with the specified data.
        /// </summary>
        /// <param name="data">A 2D array of input data.</param>
        public Matrix(double[][] data) =>
            Data = data ?? throw new ArgumentNullException(nameof(data));

        public double this[int row, int col]
        {
            get => Data[row][col];
            set => Data[row][col] = value;
        }


        /// <summary>
        /// Adds two matrices element-wise and returns the result as a new matrix.
        /// </summary>
        /// <param name="other">The matrix to be added to the current matrix.</param>
        /// <returns>A new matrix that is the element-wise sum of the two matrices.</returns>
        public Matrix Add(Matrix other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            if (Rows != other.Rows || Columns != other.Columns)
            {
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));
            }

            Matrix result = new Matrix(Rows, Columns);

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[i, j] = this[i, j] + other[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Subtracts another matrix element-wise from the current matrix and returns the result as a new matrix.
        /// </summary>
        /// <param name="other">The matrix to be subtracted from the current matrix.</param>
        /// <returns>A new matrix that is the element-wise difference of the two matrices.</returns>
        public Matrix Subtract(Matrix other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            if (Rows != other.Rows || Columns != other.Columns)
            {
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));
            }

            Matrix result = new Matrix(Rows, Columns);

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[i, j] = this[i, j] - other[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplies the current matrix with another matrix and returns the result as a new matrix.
        /// </summary>
        /// <param name="other">The matrix to be multiplied with the current matrix.</param>
        /// <returns>A new matrix that is the product of the two matrices.</returns>
        public Matrix Multiply(Matrix other)
        {
            if (other == null)
            {
                throw new ArgumentNullException(nameof(other));
            }

            if (Columns != other.Rows)
            {
                throw new ArgumentException("The number of columns in the first matrix must equal the number of rows in the second matrix.", nameof(other));
            }

            int rows = Rows;
            int columns = other.Columns;
            int common = Columns;

            Matrix result = new Matrix(rows, columns);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    double sum = 0;

                    for (int k = 0; k < common; k++)
                    {
                        sum += this[i, k] * other[k, j];
                    }

                    result[i, j] = sum;
                }
            }

            return result;
        }

        /// <summary>
        /// Transposes the current matrix and returns the result as a new matrix.
        /// </summary>
        /// <returns>A new matrix that is the transpose of the current matrix.</returns>
        public Matrix Transpose()
        {
            Matrix result = new Matrix(Columns, Rows);

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[j, i] = this[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the dot product of the current matrix and another matrix.
        /// </summary>
        /// <param name="other">The matrix to calculate the dot product with.</param>
        /// <returns>The dot product of the two matrices.</returns>
        public double DotProduct(Matrix other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            if (Rows != other.Rows || Columns != other.Columns)
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));

            double dotProduct = 0;

            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    dotProduct += this[i, j] * other[i, j];

            return dotProduct;
        }

        public static Matrix operator +(Matrix a, Matrix b) => a.Add(b);

        public static Matrix operator -(Matrix a, Matrix b) => a.Subtract(b);

        public static Matrix operator *(Matrix a, Matrix b) => a.Multiply(b);

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    sb.Append(this[i, j]);

                    if (j < Columns - 1)
                    {
                        sb.Append(", ");
                    }
                }

                if (i < Rows - 1)
                {
                    sb.AppendLine();
                }
            }

            return sb.ToString();
        }
    }

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

            if (matrix.Rows != matrix.Columns)
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
            int n = A.Rows;
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
        /// Computes the eigenvalues and eigenvectors of a square matrix using the Jacobi method.
        /// </summary>
        /// <param name="A">The square matrix to decompose.</param>
        private void ComputeEigenvaluesJacobi(Matrix A)
        {
            int n = A.Rows;
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
            int n = A.Rows;
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
