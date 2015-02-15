
#ifndef CORE_MATH_MATRIX2X2_H_
#define CORE_MATH_MATRIX2X2_H_

namespace ml {

template <class FloatType>
class DenseMatrix;
template <class FloatType>
struct EigenSystem;

//! This class provides functions to handle 3-dimensional matrices
/*! The arrangement of the matrix is row-like.
    The index of a specific position is:
    <pre>
       0  1
       2  3
    </pre>
*/
template <class FloatType> class Matrix2x2 : public BinaryDataSerialize< Matrix2x2<FloatType> >
{
public:
	//! An uninitialized matrix
	Matrix2x2() {}

	//! Initialize with values stored in an array
	Matrix2x2(const FloatType* values) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] = values[i];
		}
	}

	//! Initialize from 3 row vectors
	Matrix2x2(const point2d<FloatType>& v0, const point2d<FloatType>& v1) {
		matrix[0] = v0.x;	matrix[1] = v0.y;
		matrix[2] = v1.x;	matrix[3] = v1.y;
	}

	//! Initializes the matrix row wise
	Matrix2x2(	const FloatType &m00, const FloatType &m01,
				const FloatType &m10, const FloatType &m11) 
	{
		_m00 = m00;	_m01 = m01;
		_m10 = m10;	_m11 = m11;
	}

	//! Initialize with a matrix from another type
	template<class U>
	Matrix2x2(const Matrix2x2<U>& other) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] = (FloatType)other.getRawData()[i];
		}
	}

	//! Access element of Matrix at row x and column y for constant access
	inline FloatType at(unsigned char x, unsigned char y) const {
		assert((x<2)&&(y<2)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}
	//! Access element of Matrix at row x and column y
	inline  FloatType& at(unsigned char x, unsigned char y) {
		assert((x<2)&&(y<2)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}

	//! Access i,j-th row of Matrix for constant access
	inline  FloatType operator() (unsigned int i, unsigned int j) const {
		assert(i<2 && j<2);
		return matrix2[i][j];
	}
	//! Access i,j-th element of Matrix
	inline  FloatType& operator() (unsigned int i, unsigned int j) {
		assert(i<2 && j<2);
		return matrix2[i][j]; 
	}

	//! Access i-th element of the Matrix for constant access
	inline  FloatType operator[] (unsigned int i) const {
		assert(i<4);
		return matrix[i];
	}
	//! Access i-th element of the Matrix
	inline  FloatType& operator[] (unsigned int i) {
		assert(i<4);
		return matrix[i];
	}


	//! overwrite the matrix with an identity-matrix
	void setIdentity() {
		setScale(1.0, 1.0);
	}
	static Matrix2x2 identity() {
		Matrix2x2 res;	res.setIdentity();
		return res;
	}

	//! sets the matrix zero (or a specified value)
	void setZero(FloatType v = (FloatType)0) {
		matrix[0] = matrix[1] = v;
		matrix[2] = matrix[3] = v;
	}
	static Matrix2x2 zero(FloatType v = (FloatType)0) {
		Matrix2x2 res; res.setZero(v);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType x, FloatType y) {
		matrix[0] =   x; matrix[1] = 0.0; 
		matrix[2] = 0.0; matrix[3] =   y;
	}
	static Matrix2x2 scale(FloatType x, FloatType y) {
		Matrix2x2 res;	res.setScale(x,y);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType s) {
		setScale(s,s);
	}
	static Matrix2x2 scale(FloatType s) {
		Matrix2x2 res;	res.setScale(s);
		return res;
	}
	
	//! overwrite the matrix with a scale-matrix
	void setScale(const point2d<FloatType>& v) {
		matrix[0] = v.x; matrix[1] = 0.0;
		matrix[2] = 0.0; matrix[3] = v.y;
	}
	static Matrix2x2 scale(const point2d<FloatType>& v) {
		Matrix2x2 res;	res.setScale(v);
		return res;
	}
	
	
	//! overwrite the matrix with a diagonal matrix
	void setDiag(FloatType x, FloatType y) {
		setScale(x,y);
	}
	static Matrix2x2 diag(FloatType x, FloatType y) {
		Matrix2x2 res;	res.setDiag(x,y);
		return res;
	}
	FloatType trace() const {
		return (matrix2[0][0] + matrix2[1][1]);
	}




	//! return the product of the operand with matrix
	Matrix2x2 operator* (const Matrix2x2& other) const {
		Matrix2x2<FloatType> result;
		//TODO unroll the loop
		for (unsigned char i = 0; i < 2; i++) {
			for (unsigned char j = 0; j < 2; j++) {
				result.at(i,j) = 
					this->at(i,0) * other.at(0,j) + 
					this->at(i,1) * other.at(1,j);
			}
		}
		return result;
	}
	//! multiply operand with matrix b
	Matrix2x2& operator*= (const Matrix2x2& other) {
		Matrix2x2<FloatType> prod = (*this)*other;
		*this = prod;
		return *this;
	}
	//! multiply each element in the matrix with a scalar factor
	Matrix2x2 operator* (FloatType r) const {
		Matrix2x2<FloatType> result;
		for (unsigned int i = 0; i < 4; i++) {
			result.matrix[i] = matrix[i] * r;
		}
		return result;
	}
	//! multiply each element in the matrix with a scalar factor
	Matrix2x2& operator*= (FloatType r) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] *= r;
		}
		return *this;
	}
	//! divide the matrix by a scalar factor
	Matrix2x2 operator/ (FloatType r) const {
		Matrix2x2<FloatType> result;
		for (unsigned int i = 0; i < 4; i++) {
			result.matrix[i] = matrix[i] / r;
		}
		return result;
	}
	//! divide each element in the matrix with a scalar factor
	Matrix2x2& operator/= (FloatType r) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] /= r;
		}
		return *this;
	}
	//! transform a 2D-vector with the matrix
	point2d<FloatType> operator* (const point2d<FloatType>& v) const {
		return point3d<FloatType>(
			matrix[0]*v[0] + matrix[1]*v[1],
			matrix[2]*v[0] + matrix[3]*v[1]
		);
	}
	//! return the sum of the operand with matrix b
	Matrix2x2 operator+ (const Matrix2x2& other) const {
		Matrix2x2<FloatType> result;
		for (unsigned int i = 0; i < 4; i++) {
			result.matrix[i] = matrix[i] + other.matrix[i];
		}
		return result;
	}

	//! add matrix other to the operand
	Matrix2x2& operator+= (const Matrix2x2& other) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] += other.matrix[i];
		}
		return *this;
	}

	//! return the difference of the operand with matrix b
	Matrix2x2 operator- (const Matrix2x2& other) const {
		Matrix2x2<FloatType> result;
		for (unsigned int i = 0; i < 4; i++) {
			result.matrix[i] = matrix[i] - other.matrix[i];
		}
		return result;
	}
	//! subtract matrix other from the operand
	Matrix2x2 operator-= (const Matrix2x2& other) {
		for (unsigned int i = 0; i < 4; i++) {
			matrix[i] -= other.matrix[i];
		}
		return *this;
	}

	//! return the determinant of the matrix
	FloatType det() const {
		return matrix[0]*matrix[3] - matrix[1]*matrix[2];
	}


	//! get the x column out of the matrix
	point2d<FloatType> xcol() const {
		return point2d<FloatType>(matrix[0],matrix[2]);
	}
	//! get the y column out of the matrix
	point3d<FloatType> ycol() const {
		return point2d<FloatType>(matrix[1],matrix[3]);
	}
	//! get the x row out of the matrix
	point2d<FloatType> xrow() const {
		return point2d<FloatType>(matrix[0],matrix[1]);
	}
	//! get the y row out of the matrix
	point2d<FloatType> yrow() const {
		return point2d<FloatType>(matrix[2],matrix[3]);
	}

	//! return the inverse matrix; but does not change the current matrix
	Matrix2x2 getInverse() const {
		FloatType inv[4];

		inv[0] = matrix[3];
		inv[1] = -matrix[1];
		inv[2] = -matrix[2];
		inv[3] = matrix[0];

		FloatType matrixDet = det();		
		FloatType matrixDetr= (FloatType)1.0 / matrixDet;

		Matrix2x2<FloatType> res;
		for (unsigned int i = 0; i < 4; i++) {
			res.matrix[i] = inv[i] * matrixDetr;
		}
		return res;
	}

	//! overwrite the current matrix with its inverse
	void invert() {
		*this = getInverse();
	}
	//! return the transposed matrix
	Matrix2x2 getTranspose() const {
		Matrix2x2<FloatType> result;
		for(unsigned char x = 0; x < 2; x++) {
			result.at(x,0) = at(0,x);
			result.at(x,1) = at(1,x);
		}
		return result;
	}
	//! transpose the matrix in place
	void transpose() {
		*this = getTranspose();
	}

	FloatType* getRawData()
	{
		return matrix;
	}

	//! computes the tensor product between two vectors
	static Matrix2x2 tensorProduct(const point2d<FloatType> &v0, const point2d<FloatType> &v1) {
		Matrix2x2 ret;
		ret._m00 = v0.x * v1.x;		ret._m01 = v0.x * v1.y;
		ret._m10 = v0.y * v1.x;		ret._m11 = v0.y * v1.y;
		return ret;
	}

	unsigned int rank(FloatType eps = (FloatType)0.00001) const {
		return util::rank<Matrix2x2<FloatType>, FloatType>(*this, 2, eps);
	}

	EigenSystem<FloatType> eigenSystem() const {
		return EigenSolver<FloatType>::solve<EigenSolver<FloatType>::TYPE_DEFAULT>(*this);
	}

protected:

private:
	union {
		//! access matrix using a single array
		FloatType matrix[4];
		//! access matrix using a two-dimensional array
		FloatType matrix2[2][2];
		//! access matrix using single elements
		struct { 
			FloatType
				_m00, _m01,
				_m10, _m11;
		};
	};
};

//! writes to a stream
template <class FloatType> 
inline std::ostream& operator<<(std::ostream& s, const Matrix2x2<FloatType>& m)
{ 
	return (
		s << 
		m(0,0) << " " << m(0,1) << " " << std::endl <<
		m(1,0) << " " << m(1,1) << " " << std::endl
		);
}

//! reads from a stream
template <class FloatType> 
inline std::istream& operator>>(std::istream& s, const Matrix2x2<FloatType>& m)
{ 
	return (
		s >> 
		m(0,0) >> m(0,1) >>
		m(1,0) >> m(1,1)
		);
}


typedef Matrix2x2<int> mat2i;
typedef Matrix2x2<int> mat2u;
typedef Matrix2x2<float> mat2f;
typedef Matrix2x2<double> mat2d;

}  // namespace ml

#endif  // CORE_MATH_MATRIX2X2_H_
