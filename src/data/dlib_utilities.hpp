//
// Created by asem on 14/08/18.
//

#ifndef MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP
#define MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP

#include <dlib/matrix.h>

template<typename T>
struct op_vector_to_matrix
{
    /*!
        This object defines a matrix expression that holds a reference to a std::vector<T>
        and makes it look like a column vector.  Thus it enables you to use a std::vector
        as if it was a dlib::matrix.
    !*/
    op_vector_to_matrix( const std::vector<T> &vect_ ) : vect( vect_ )
    {}

    const std::vector<T> &vect;

    // This expression wraps direct memory accesses so we use the lowest possible cost.
    const static long cost = 1;

    const static long NR = 0; // We don't know the length of the vector until runtime.  So we put 0 here.
    const static long NC = 1; // We do know that it only has one column (since it's a vector)
    typedef T type;
    // Since the std::vector doesn't use a dlib memory manager we list the default one here.
    typedef dlib::default_memory_manager mem_manager_type;
    // The layout type also doesn't really matter in this case.  So we list row_major_layout
    // since it is a good default.
    typedef dlib::row_major_layout layout_type;

    // Note that we define const_ret_type to be a reference type.  This way we can
    // return the contents of the std::vector by reference.
    typedef const T &const_ret_type;

    const_ret_type apply( long r, long ) const
    { return vect[r]; }

    long nr() const
    { return vect.size(); }

    long nc() const
    { return 1; }

    // This expression never aliases anything since it doesn't contain any matrix expression (it
    // contains only a std::vector which doesn't count since you can't assign a matrix expression
    // to a std::vector object).
    template<typename U>
    bool aliases( const dlib::matrix_exp<U> & ) const
    { return false; }

    template<typename U>
    bool destructively_aliases( const dlib::matrix_exp<U> & ) const
    { return false; }
};

template<typename T>
const dlib::matrix_op<op_vector_to_matrix<T> > vector_to_matrix( const std::vector<T> &vector )
{
    typedef op_vector_to_matrix<T> op;
    return dlib::matrix_op<op>( op( vector ));
}

#endif //MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP
