/**
 * Copyright (c) 2014-2015 Patrick Wieschollek
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

#ifndef ISOLVER_H_
#define ISOLVER_H_


#include <functional>
#include "Meta.h"

namespace pwie
{



    class ISolver
    {

        public:
            Options settings;
            ISolver();
            virtual ~ISolver();

            void solve( Vector & x0,
                        function_t const & FunctionValue,
                        gradient_t const & FunctionGradient = std::function<void(const Eigen::VectorXd & x, Eigen::VectorXd & grad)>(),
                        hessian_t  const & FunctionHessian = EMPTY_HESSIAN );

            virtual void internalSolve( Vector & x0,
                                        function_t const & FunctionValue,
                                        gradient_t const & FunctionGradient,
                                        hessian_t  const & FunctionHessian = EMPTY_HESSIAN) = 0;

            static double linesearch( Vector const & x,
                                      Vector const & direction,
                                      function_t const & FunctionValue,
                                      gradient_t const & FunctionGradient );

            static double linesearch( Vector const & x,
                                      Vector const & direction,
                                      Eigen::MatrixXd const & hessian,
                                      function_t const & FunctionValue,
                                      gradient_t const & FunctionGradient );

    };

} /* namespace pwie */

#endif /* ISOLVER_H_ */
