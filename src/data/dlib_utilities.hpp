//
// Created by asem on 14/08/18.
//

#ifndef MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP
#define MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP

#include <dlib/matrix.h>

namespace dlib_utilities {

template<typename T>
struct column_matrix_like
{
    /*!
        This object defines a matrix expression that holds a reference to a std::vector<T>
        and makes it look like a column vector.  Thus it enables you to use a std::vector
        as if it was a dlib::matrix.
    !*/
    explicit column_matrix_like( std::vector<T> vect_ ) : _vect( std::move( vect_ ))
    {}

    template<typename MatrixExpression>
    explicit column_matrix_like( MatrixExpression exp )
    {
        _vect.reserve( exp.size());
        _vect.insert( _vect.end(), exp.begin(), exp.end());
    }

    template<typename MatrixExpression>
    column_matrix_like &operator=( MatrixExpression exp )
    {
        _vect.clear();
        _vect.reserve( exp.nr());
        _vect.insert( _vect.end(), exp.cbegin(), exp.cend());
        return *this;
    }

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

    const_ret_type apply(
            long r,
            long
    ) const
    { return _vect[r]; }

    long nr() const
    { return _vect.size(); }

    long nc() const
    { return 1; }

    std::vector<T> steal_vector()
    {
        return std::move( _vect );
    }

    // This expression never aliases anything since it doesn't contain any matrix expression (it
    // contains only a std::vector which doesn't count since you can't assign a matrix expression
    // to a std::vector object).
    template<typename U>
    bool aliases( const dlib::matrix_exp<U> & ) const
    { return false; }

    template<typename U>
    bool destructively_aliases( const dlib::matrix_exp<U> & ) const
    { return false; }

private:
    std::vector<T> _vect;
};

template<typename T>
struct row_matrix_like
{
    /*!
        This object defines a matrix expression that holds a reference to a std::vector<T>
        and makes it look like a column vector.  Thus it enables you to use a std::vector
        as if it was a dlib::matrix.
    !*/
    explicit row_matrix_like( std::vector<T> vect_ ) : _vect( std::move( vect_ ))
    {}

    template<typename MatrixExpression>
    explicit row_matrix_like( MatrixExpression exp )
    {
        _vect.reserve( exp.size());
        _vect.insert( _vect.end(), exp.begin(), exp.end());
    }

    template<typename MatrixExpression>
    row_matrix_like &operator=( MatrixExpression exp )
    {
        _vect.clear();
        _vect.reserve( exp.size());
        _vect.insert( _vect.end(), exp.cbegin(), exp.cend());
        return *this;
    }

    // This expression wraps direct memory accesses so we use the lowest possible cost.
    const static long cost = 1;

    const static long NR = 1; // We don't know the length of the vector until runtime.  So we put 0 here.
    const static long NC = 0; // We do know that it only has one column (since it's a vector)
    typedef T type;
    // Since the std::vector doesn't use a dlib memory manager we list the default one here.
    typedef dlib::default_memory_manager mem_manager_type;
    // The layout type also doesn't really matter in this case.  So we list row_major_layout
    // since it is a good default.
    typedef dlib::row_major_layout layout_type;

    // Note that we define const_ret_type to be a reference type.  This way we can
    // return the contents of the std::vector by reference.
    typedef const T &const_ret_type;

    const_ret_type apply(
            long ,
            long c
    ) const
    { return _vect[c]; }

    long nr() const
    { return 1; }

    long nc() const
    { return _vect.size(); }

    std::vector<T> steal_vector()
    {
        return std::move( _vect );
    }

    // This expression never aliases anything since it doesn't contain any matrix expression (it
    // contains only a std::vector which doesn't count since you can't assign a matrix expression
    // to a std::vector object).
    template<typename U>
    bool aliases( const dlib::matrix_exp<U> & ) const
    { return false; }

    template<typename U>
    bool destructively_aliases( const dlib::matrix_exp<U> & ) const
    { return false; }

private:
    std::vector<T> _vect;
};

template<typename InputContainer>
auto vector_to_column_matrix_like( InputContainer &&container )
{
    using T = typename std::remove_reference_t<InputContainer>::value_type;
    using MatrixLike = column_matrix_like<T>;

    std::vector<T> delegate = std::forward<InputContainer>( container );
    return dlib::matrix_op<MatrixLike>( MatrixLike( std::move( delegate )));
}

template<typename InputContainer>
auto vector_to_row_matrix_like( InputContainer &&container )
{
    using T = typename std::remove_reference_t<InputContainer>::value_type;
    using MatrixLike = row_matrix_like<T>;

    std::vector<T> delegate = std::forward<InputContainer>( container );
    return dlib::matrix_op<MatrixLike>( MatrixLike( std::move( delegate )));
}

}
#endif //MARKOVIAN_FEATURES_DLIB_UTILITIES_HPP
