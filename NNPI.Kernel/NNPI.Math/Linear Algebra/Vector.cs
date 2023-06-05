using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NNPI.Kernel.NNPIMath.Linear_Algebra
{
    /// <summary>
    /// Represents a mathematical vector and provides basic operations.
    /// </summary>
    public class Vector : IEnumerable<double>
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

        /// <summary>
        /// Creates a vector with the specified size and sets the element at the specified index to 1.
        /// </summary>
        /// <param name="size">The size of the resulting vector.</param>
        /// <param name="index">The index of the element to set to 1.</param>
        /// <returns>A vector with the specified size and single element set to 1.</returns>
        public static Vector Basis(int size, int index)
        {
            if (size < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Size must be greater than 0.");
            }

            if (index < 0 || index >= size)
            {
                throw new ArgumentOutOfRangeException(nameof(index), "Invalid index.");
            }

            Vector basis = new Vector(size);
            basis[index] = 1;
            return basis;
        }

        /// <summary>
        /// Returns a subvector that starts at the specified index.
        /// </summary>
        /// <param name="startIndex">The starting index.</param>
        /// <param name="length">The length of the subvector.</param>
        /// <returns>A new vector containing the specified subvector.</returns>
        public Vector SubVector(int startIndex, int length)
        {
            if (startIndex < 0 || startIndex + length > Length)
            {
                throw new ArgumentOutOfRangeException("Invalid index for extracting subvector.");
            }

            Vector subVector = new Vector(length);
            for (int i = 0; i < length; i++)
            {
                subVector[i] = this[startIndex + i];
            }
            return subVector;
        }

        /// <summary>
        /// Computes the Euclidean norm (magnitude) of the vector.
        /// </summary>
        /// <returns>The Euclidean norm of the vector.</returns>
        public double Norm()
        {
            double sum = 0;
            for (int i = 0; i < Length; i++)
            {
                sum += this[i] * this[i];
            }
            return Math.Sqrt(sum);
        }


        /// <summary>
        /// Computes the outer product of two vectors.
        /// </summary>
        /// <param name="v">The second vector.</param>
        /// <returns>The resulting outer product as a matrix.</returns>
        public Matrix OuterProduct(Vector v)
        {
            if (v == null)
            {
                throw new ArgumentNullException(nameof(v));
            }

            Matrix result = new Matrix(Length, v.Length);
            for (int i = 0; i < Length; i++)
            {
                for (int j = 0; j < v.Length; j++)
                {
                    result[i, j] = this[i] * v[j];
                }
            }
            return result;
        }

        /// <summary>
        /// Computes the dot product of two vectors.
        /// </summary>
        /// <param name="v">The second vector.</param>
        /// <returns>The resulting dot product as a scalar value.</returns>
        public double DotProduct(Vector v)
        {
            if (v == null)
            {
                throw new ArgumentNullException(nameof(v));
            }

            if (Length != v.Length)
            {
                throw new ArgumentException("Vectors must have the same length.", nameof(v));
            }

            double result = 0;
            for (int i = 0; i < Length; i++)
            {
                result += this[i] * v[i];
            }
            return result;
        }

        /// <summary>
        /// Returns the subvector of this vector that starts at the specified index.
        /// </summary>
        /// <param name="startIndex">The starting index of the subvector. Must be non-negative and less than the vector length.</param>
        /// <returns>A new Vector instance representing the subvector.</returns>
        public Vector SubVector(int startIndex)
        {
            if (startIndex < 0 || startIndex >= Length)
            {
                throw new ArgumentException("Invalid start index for subvector extraction.", nameof(startIndex));
            }

            int subVectorLength = Length - startIndex;
            double[] subVectorData = new double[subVectorLength];
            Array.Copy(Data, startIndex, subVectorData, 0, subVectorLength);

            return new Vector(subVectorData);
        }

        public static Vector operator +(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] + scalar;
            }
            return result;
        }

        public static Vector operator -(Vector vector, double scalar) => vector + (-scalar);

        public static Vector operator *(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] * scalar;
            }
            return result;
        }

        public static Vector operator /(Vector vector, double scalar)
        {
            if (scalar == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero.");
            }
            return vector * (1 / scalar);
        }

        public static Vector operator +(double scalar, Vector vector) => vector + scalar;

        public static Vector operator -(Vector vector)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = -vector[i];
            }
            return result;
        }

        public static Vector operator -(double scalar, Vector vector) => (-vector) + scalar;

        public static Vector operator *(double scalar, Vector vector) => vector * scalar;

        public static Vector operator /(double scalar, Vector vector) => (1 / scalar) * vector;

        public static bool operator <(Vector a, Vector b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Vectors must have the same length for comparison.");
            }

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] >= b[i])
                {
                    return false;
                }
            }
            return true;
        }

        public static bool operator >(Vector a, Vector b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Vectors must have the same length for comparison.");
            }

            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] <= b[i])
                {
                    return false;
                }
            }
            return true;
        }

        public static bool operator <(Vector vector, int scalar)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] >= scalar)
                {
                    return false;
                }
            }
            return true;
        }

        public static bool operator >(Vector vector, int scalar)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] <= scalar)
                {
                    return false;
                }
            }
            return true;
        }

        public double Mean()
        {
            double sum = 0;
            int length = Data.Length;
            for (int i = 0; i < length; i++)
            {
                sum += Data[i];
            }
            return sum / length;
        }

        public IEnumerator<double> GetEnumerator() => ((IEnumerable<double>)Data).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
