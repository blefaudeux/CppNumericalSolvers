/**
* Copyright (c) 2014-2015 Benjamin Lefaudeux
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef LEVENBERGMARQUARDT_H_
#define LEVENBERGMARQUARDT_H_

#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "isolver.h"
#include "../problem.h"

// Benjamin : just a wrapper around the Eigen solver, so that Levenberg Marquardt
// can be tested against all the other solvers

namespace cppoptlib
{
    // Generic functor
    // Reusing Eigen/unsupported/test file as an example
    template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
    struct Functor
    {
            typedef _Scalar Scalar;
            enum {
                InputsAtCompileTime = NX,
                ValuesAtCompileTime = NY
            };
            typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
            typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
            typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

            const int m_inputs, m_values;

            Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
            Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

            int inputs() const { return m_inputs; }
            int values() const { return m_values; }

            // you should define that in the subclass :
            //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
    };

    template <typename T>
    class LevenbergMarquardtSolver : public ISolver<T, 0>
    {
            using Vec = Eigen::Matrix< T,Eigen::Dynamic, 1>;
            using Mat = Eigen::Matrix< T,Eigen::Dynamic, Eigen::Dynamic>;

        public:
            void minimize( Problem<T> &objFunc, Vector<T> & x)
            {
                // Build an adhoc functor from scratch
                struct adhoc_functor : Functor<T>
                {
                        adhoc_functor(Problem<T> &objFunc, int dim):
                            Functor<T>(dim, 1),
                            m_objFunc(objFunc)
                        {}

                        int operator()( Vec const & x, Vec & fvec) const
                        {
                            fvec = Vec::Zero(1);
                            fvec(0) = m_objFunc.value(x);
                            return 0;
                        }

                        int df(Vec const & x, Mat & fjac) const
                        {
                            Vec grad;
                            m_objFunc.gradient(x, grad);
                            fjac = grad.asDiagonal(); // This is not really correct.. trying
                            return 0;
                        }

                        Problem<T> & m_objFunc;
                };

                // Use the Eigen solver under the hood
                adhoc_functor functor(objFunc, x.rows());
                Eigen::LevenbergMarquardt<adhoc_functor> lm(functor);

                int const retVal = lm.minimize(x);
                std::cout << "Final parameters : " << x << std::endl;
                std::cout << "Final value : " << lm.fnorm;
                std::cout << "Iterations : " << lm.iter;
            }
    };

} /* namespace cppoptlib */

#endif /* LEVENBERGMARQUARDT_H_ */
