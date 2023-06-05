using NNPI.Kernel.NNPIMath.Linear_Algebra;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNPI.Kernel.NNPI.Math
{
    public static class MatrixExtensions
    {
        /// <summary>
        /// Converts a Matrix<double> to a jagged 2D array of doubles, organized by rows.
        /// </summary>
        /// <param name="matrix">The matrix to convert.</param>
        /// <returns>A jagged 2D array of doubles representing the matrix.</returns>
        public static double[][] ToRowArrays(this Matrix matrix)
        {
            int numRows = matrix.RowCount;
            int numCols = matrix.ColumnCount;
            double[][] rowArrays = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                rowArrays[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    rowArrays[i][j] = matrix[i, j];
                }
            }

            return rowArrays;
        }
    }
}
