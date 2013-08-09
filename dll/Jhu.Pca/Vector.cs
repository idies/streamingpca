using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Jhu.Pca
{
    public class Vector
    {
        private double[] value;
        private double[] weight;
        private bool[] mask;

        public double[] Value
        {
            get { return this.value; }
            set { this.value = value; }
        }

        public double[] Weight
        {
            get { return this.weight; }
            set { this.weight = value; }
        }

        public bool[] Mask
        {
            get { return this.mask; }
            set { this.mask = value; }
        }

        public Vector()
        {
            InitializeMembers();
        }

        public Vector(double[] value)
        {
            InitializeMembers();

            this.value = value;
        }

        public Vector(double[] value, double[] weight, bool[] mask)
        {
            InitializeMembers();

            this.value = value;
            this.weight = weight;
            this.mask = mask;
        }

        private void InitializeMembers()
        {
            this.value = null;
            this.weight = null;
            this.mask = null;
        }
    }
}
