using System.Text;

namespace NNPI.Kernel.NNPIMath.Linear_Algebra
{
    /// <summary>
    /// Represents a mathematical matrix and provides basic operations.
    /// </summary>
    public class Matrix
    {
        public double[][] Data { get; }
        public int RowCount => Data.Length;
        public int ColumnCount => Data[0].Length;

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

        /// <summary>
        /// Gets the element at the specified row and column.
        /// </summary>
        /// <param name="row">The row index of the element.</param>
        /// <param name="col">The column index of the element.</param>
        /// <returns>The element at the specified row and column.</returns>
        public double this[int row, int col]
        {
            get
            {
                if (row < 0 || row >= RowCount || col < 0 || col >= ColumnCount)
                {
                    throw new ArgumentOutOfRangeException("Invalid row or column index.");
                }

                return Data[row][col];
            }
            set
            {
                if (row < 0 || row >= RowCount || col < 0 || col >= ColumnCount)
                {
                    throw new ArgumentOutOfRangeException("Invalid row or column index.");
                }

                Data[row][col] = value;
            }
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

            if (RowCount != other.RowCount || ColumnCount != other.ColumnCount)
            {
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));
            }

            Matrix result = new Matrix(RowCount, ColumnCount);

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
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

            if (RowCount != other.RowCount || ColumnCount != other.ColumnCount)
            {
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));
            }

            Matrix result = new Matrix(RowCount, ColumnCount);

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
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

            if (ColumnCount != other.RowCount)
            {
                throw new ArgumentException("The number of columns in the first matrix must equal the number of rows in the second matrix.", nameof(other));
            }

            int rows = RowCount;
            int columns = other.ColumnCount;
            int common = ColumnCount;

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
            Matrix result = new Matrix(ColumnCount, RowCount);

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
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

            if (RowCount != other.RowCount || ColumnCount != other.ColumnCount)
                throw new ArgumentException("Both matrices must have the same dimensions.", nameof(other));

            double dotProduct = 0;

            for (int i = 0; i < RowCount; i++)
                for (int j = 0; j < ColumnCount; j++)
                    dotProduct += this[i, j] * other[i, j];

            return dotProduct;
        }

        public static Matrix operator +(Matrix a, Matrix b) => a.Add(b);

        public static Matrix operator -(Matrix a, Matrix b) => a.Subtract(b);

        public static Matrix operator *(Matrix a, Matrix b) => a.Multiply(b);

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
                {
                    sb.Append(this[i, j]);

                    if (j < ColumnCount - 1)
                    {
                        sb.Append(", ");
                    }
                }

                if (i < RowCount - 1)
                {
                    sb.AppendLine();
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Creates a deep copy of the current matrix.
        /// </summary>
        /// <returns>A new matrix with the same elements as the current matrix.</returns>
        public Matrix Copy()
        {
            Matrix copy = new Matrix(RowCount, ColumnCount);
            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
                {
                    copy[i, j] = this[i, j];
                }
            }
            return copy;
        }

        /// <summary>
        /// Creates an identity matrix with the specified size.
        /// </summary>
        /// <param name="n">The number of rows and columns in the identity matrix.</param>
        /// <returns>An identity matrix of size n x n.</returns>
        public static Matrix Identity(int n)
        {
            Matrix identity = new Matrix(n, n);
            for (int i = 0; i < n; i++)
            {
                identity[i, i] = 1;
            }
            return identity;
        }

        /// <summary>
        /// Returns the specified column as a Vector.
        /// </summary>
        /// <param name="columnIndex">The index of the column to retrieve.</param>
        /// <returns>The specified column as a Vector.</returns>
        public Vector GetColumn(int columnIndex)
        {
            if (columnIndex < 0 || columnIndex >= ColumnCount)
            {
                throw new ArgumentOutOfRangeException(nameof(columnIndex), "Invalid column index.");
            }

            Vector column = new Vector(RowCount);
            for (int i = 0; i < RowCount; i++)
            {
                column[i] = this[i, columnIndex];
            }
            return column;
        }

        /// <summary>
        /// Sets the specified submatrix starting at the specified row and column.
        /// </summary>
        /// <param name="row">The starting row index.</param>
        /// <param name="column">The starting column index.</param>
        /// <param name="subMatrix">The submatrix to set.</param>
        /// <returns>The updated matrix with the specified submatrix set.</returns>
        public Matrix SetSubMatrix(int row, int column, Matrix subMatrix)
        {
            if (subMatrix == null)
            {
                throw new ArgumentNullException(nameof(subMatrix));
            }

            if (row < 0 || row + subMatrix.RowCount > RowCount || column < 0 || column + subMatrix.ColumnCount > ColumnCount)
            {
                throw new ArgumentOutOfRangeException("Invalid row or column index for setting submatrix.");
            }

            for (int i = 0; i < subMatrix.RowCount; i++)
            {
                for (int j = 0; j < subMatrix.ColumnCount; j++)
                {
                    this[row + i, column + j] = subMatrix[i, j];
                }
            }
            return this;
        }

        /// <summary>
        /// Divides each element of the matrix by the specified scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to divide the matrix elements by.</param>
        /// <returns>A new Matrix instance with the divided elements.</returns>
        public Matrix DivideByScalar(double scalar)
        {
            if (scalar == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero.");
            }

            Matrix result = new Matrix(RowCount, ColumnCount);

            for (int i = 0; i < RowCount; i++)
            {
                for (int j = 0; j < ColumnCount; j++)
                {
                    result.Data[i][j] = Data[i][j] / scalar;
                }
            }

            return result;
        }

        public static Matrix operator +(Matrix matrix, double scalar)
        {
            Matrix result = new Matrix(matrix.RowCount, matrix.ColumnCount);
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    result.Data[i][j] = matrix.Data[i][j] + scalar;
                }
            }
            return result;
        }

        public static Matrix operator -(Matrix matrix, double scalar)
        {
            return matrix + (-scalar);
        }

        public static Matrix operator *(Matrix matrix, double scalar)
        {
            Matrix result = new Matrix(matrix.RowCount, matrix.ColumnCount);
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    result.Data[i][j] = matrix.Data[i][j] * scalar;
                }
            }
            return result;
        }
        public static Matrix operator *(double scalar, Matrix matrix) => matrix * scalar;

        public static Matrix operator /(Matrix matrix, double scalar)
        {
            if (scalar == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero.");
            }
            return matrix * (1 / scalar);
        }

        public static Matrix operator *(Matrix matrix, int scalar) => matrix * (double)scalar;
        public static Matrix operator *(int scalar, Matrix matrix) => matrix * (double)scalar;

    }
}
