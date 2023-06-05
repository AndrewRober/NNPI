﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNPI.Kernel.Weigh_tInitializers
{
    public abstract class WeightInitializer
    {
        /// <summary>
        /// Initializes the weights of a given matrix.
        /// </summary>
        /// <param name="weights">The weights matrix to initialize.</param>
        public abstract void Initialize(double[,] weights);
    }


}
