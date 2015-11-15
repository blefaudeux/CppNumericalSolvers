CppOptimizationLibrary
=================================================================

[![Build Status](https://api.travis-ci.org/PatWie/CppNumericalSolvers.svg?branch=master)](http://travis-ci.org/PatWie/CppNumericalSolvers)
[![Build Status](https://img.shields.io/github/release/PatWie/CppNumericalSolvers.svg)](https://github.com/PatWie/CppNumericalSolvers/releases)
[![Build Status](https://img.shields.io/github/issues/PatWie/CppNumericalSolvers.svg)](https://github.com/PatWie/CppNumericalSolvers/issues)


Install
-----------

Before compiling you need to adjust some paths. All dependencies (currently just [Eigen3][eigen3]) can by downloaded by a single bashscript.

    ./get_dependencies.sh

To compile the examples and the unit test just do

    # copy configuration file
    cp CppNumericalSolvers.config.example CppNumericalSolvers.config
    # adjust paths and compiler (supports g++, clang++)
    edit CppNumericalSolvers.config
    # create a new directory
    mkdir build && cd build   
    cmake ..
    # build tests and demo  
    make all    
    # run all tests                
    ./bin/verify  
    # run an example
    ./bin/examples/linearregression    

For using the MATLAB-binding run `make.m` inside the MATLAB folder once.

Extensive Introduction
-----------

This repository contains several solvers implemented in modern C++11 using the [Eigen3][eigen3] library for efficient matrix computations (SSE instructions, vectorization). All implementations are written from scratch. You can use this library in **C++ and [Matlab][matlab]** in an comfortable way.  All solvers are tested on challenging objective functions by unit tests using the Google Testing Framework.

The library currently contains the following solvers:

- gradient descent solver
- conjugate gradient descent solver
- Newton descent solver
- BFGS solver
- L-BFGS solver
- L-BFGS-B solver

You can use all these solvers directly from MATLAB.
The benchmark-file contains the following test functions:

*Rosenbrock, Beale, GoldsteinPrice, Booth, Matyas, Levi*

Further test-functions are planned.

For checking your gradient this library use high-order central difference.

# Usage 
There are currently two ways to use this library: directly in your C++ code or in MATLAB by calling the provided mex-File.

## C++ 

There are several examples within the `src/examples` directory. These are build into `buil/bin/examples` during `make all`.
Checkout `rosenbrock.cpp`. Your objective and gradient computations should be stored into a tiny class. The most simple usage is

    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
    }

In contrast to previous versions of this library, I switched to classes instead of lambda function. If you poke the examples, you will notice that this much easier to write and understand. The only method a problem has to provide is the `value` memeber, which returns the value of the objective function.
For most solvers it should be useful to implement the gradient computation, too. Otherwise the library internally will uses finite difference for gradient computations (which is definetely unstable and slow!).

    class YourProblem : public Problem<double> {
      double value(const Vector<double> &x) {}
      void gradient(const Vector<double> &x, Vector<double> &grad) {}
    }

Notice, the gradient is passed by reference!
After defining the problem it can be initialised in your code by:

    // init problem
    YourProblem f;
    // test your gradient ONCE
    bool probably_correct = f.checkGradient(x);

By convention, a solver minimizes a given objective function starting in `x`

    // choose a solver
    BfgsSolver<double> solver;
    // and minimize the function
    solver.minimize(f, x);
    double minValue = f(x);

For convenience there are some typedefs:

    cppoptlib::Vector<T> is a column vector Eigen::Matrix<T, Eigen::Dynamic, 1>;
    cppoptlib::Matrix<T> is a matrix        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

### full example

    class Rosenbrock : public Problem<double> {
      public:
        double value(const Vector<double> &x) {
            const double t1 = (1 - x[0]);
            const double t2 = (x[1] - x[0] * x[0]);
            return   t1 * t1 + 100 * t2 * t2;
        }
        void gradient(const Vector<double> &x, Vector<double> &grad) {
            grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
            grad[1]  = 200 * (x[1] - x[0] * x[0]);
        }
    };
    int main(int argc, char const *argv[]) {
        Rosenbrock<double> f;
        Vector<double> x(2); x << -1, 2;
        BfgsSolver<double> solver;
        solver.minimize(f, x);
        std::cout << "argmin      " << x.transpose() << std::endl;
        std::cout << "f in argmin " << f(x) << std::endl;
        return 0;
    }

### using box-constraints

The `L-BFGS-B` algorithm allows to optimize functions with box-constraints, i.e., `min_x f(x) s.t. a <= x <= b` for some `a, b`. Given a `problem`-class you can enter these constraints by

    cppoptlib::YourProblem<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));
    f.setUpperBound(Vector<double>::Ones(DIM)*5);

If you do not specify a bound, the algorithm will assume the unbounded case, eg.

    cppoptlib::YourProblem<T> f;
    f.setLowerBound(Vector<double>::Zero(DIM));

will optimize in x s.t. `0 <= x`.

## within MATLAB

Simply create a function file for the objective and replace `fminsearch` or `fminunc` with `cppoptlib`. If you want to use symbolic gradient or hessian information see file `example.m` for details. A basic example would be:

    x0 = [-1,2]';
    [fx,x] = cppoptlib(x0, @rosenbrock,'gradient',@rosenbrock_grad,'solver','bfgs');
    fx     = cppoptlib(x0, @rosenbrock);
    fx     = fminsearch(x0, @rosenbrock);

Even optimizing without any gradient information this library outperforms optimization routines from MATLAB on some problems.

# Benchmarks

Currently, not all solvers are equally good at all objective functions. The file `src/test/benchmark.cpp` contains some challenging objective functions which are tested by each provided solver. Note, MATLAB will also fail at some objective functions.

# Contribute

Make sure that `make lint` does not display any errors and check if travis is happy. Do not forget to `chmod +x lint.py`. It would be great, if some of the Forks-Owner are willing to make pull-request.

# License

    Copyright (c) 2014-2015 Patrick Wieschollek
    Copyright (c) 2015+,    the respective contributors
    Url: https://github.com/PatWie/CppNumericalSolvers

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    
This fork introduces
-----------
- Minor code style changes
- CMakeLists build chain (hopefully more cross platform)
- ... (more to come)

[eigen3]: http://eigen.tuxfamily.org/
[matlab]: http://www.mathworks.de/products/matlab/
[lastversion]: https://github.com/PatWie/CppNumericalSolvers/releases/tag/v1.0.0alpha2