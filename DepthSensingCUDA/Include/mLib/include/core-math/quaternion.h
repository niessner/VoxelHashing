#pragma once

#ifndef _CORE_MATH_QUATERNION_H_
#define _CORE_MATH_QUATERNION_H_


namespace ml {


//! Quaternions are used to describe rotations
template <class FloatType> class Quaternion {
	public:
		//! construct a quaternion that does no rotation
		Quaternion() : m_Real(1), m_Imag(0,0,0) {}

		Quaternion(FloatType r, FloatType i, FloatType j, FloatType k) {
			m_Real = r;
			m_Imag = point3d<FloatType>(i,j,k);
		}

		//! construct a quaternion explicitly
		Quaternion( FloatType real, const point3d<FloatType>& imag ) : m_Real(real), m_Imag(imag) {}

		//! construct a quaternion given a rotation-axis and an angle (in degrees)
		Quaternion( const point3d<FloatType>& axis, FloatType angle ) {
			FloatType halfAngleRad = ( FloatType ) math::PI * angle / ( FloatType ) 360.0;
			FloatType axisLength = axis.length();
			if ( axisLength > Quaternion<FloatType>::EPSILON ) {
				m_Real = cos( halfAngleRad );
				m_Imag = axis * ( sin( halfAngleRad ) / axisLength );
			} else {
				m_Real = 1;
				m_Imag = point3d<FloatType>( 0, 0, 0 );
			}
		}

		//! Constructs a quaternion between a start and end point
		Quaternion(const point3d<FloatType>& from, const point3d<FloatType>& to) {
			//point3d<FloatType> vecHalf = (from + to).getNormalized();
			//m_Real = vecHalf | to;
			//m_Imag = vecHalf ^ to;

			point3d<FloatType> v0 = from.getNormalized();
			point3d<FloatType> v1 = to.getNormalized();

			const FloatType d = v0 | v1;
			if (d >= (FloatType)1.0) // If dot == 1, vectors are the same
			{
				setIdentity();
				return;
			}
			else if (d <= (FloatType)-1.0) // exactly opposite
			{
				point3d<FloatType> axis(1.0, 0.0, 0.0);
				axis = axis ^ v0;
				if (axis.length()==0)
				{
					axis = point3d<FloatType>(0.0,1.0,0.0);
					axis = axis ^ v0;
				}
				m_Imag = axis.getNormalized();
				m_Real = 0;
				normalize();
				return;
				//return set(axis.X, axis.Y, axis.Z, 0).normalize();
			}

			const FloatType s = sqrt( (1+d)*2 ); // optimize inv_sqrt
			const FloatType invs = (FloatType)1.0 / s;
			const point3d<FloatType> c = (v0^v1)*invs;
			m_Imag = c;
			m_Real = s*(FloatType)0.5;
			normalize();
			//return set(c.X, c.Y, c.Z, s * 0.5f).normalize();
		}


		inline float sgn(float x) {return (x >= 0.0f) ? +1.0f : -1.0f;}

		Quaternion(const Matrix4x4<FloatType>& m) {
			assert(m.isAffine());
			constructFromMatrix(m.getMatrix3x3());
		}

		//! construct a quaternion based on a matrix: todo check that!
		Quaternion(const Matrix3x3<FloatType>& m) {
			constructFromMatrix(m);
		}

		//! sets the quaternion to 1,0,0,0 (i.e., no rotation)
		inline void setIdentity() {
			m_Real = 1;
			m_Imag = point3d<FloatType>(0,0,0);
		}

		//! query the real part of the quaternion
		inline FloatType real() const;
		//! query the imaginary part of the quaternion
		inline point3d<FloatType> imag() const;

		//! return quaternion as LinAl vector
		//inline doubleVec LinAlVec() const;

		//! set real part of the quaternion
		inline void setReal( const FloatType r );

		//! set imaginary part of the quaternion
		inline void setImag( const point3d<FloatType>& imag );

		//! query the axis of the rotation described by the quaternion
		point3d<FloatType> axis() const;
		//! query the angle of the rotation described by the quaternion in radians
		inline FloatType angleRad() const;
		//! query the angle of the rotation described by the quaternion in degrees
		inline FloatType angleDeg() const;
		//! query the transformation-matrix of the rotation described by the quaternion
		inline Matrix4x4<FloatType> matrix4x4() const {
			return Matrix4x4<FloatType>(matrix3x3());
		}
		inline Matrix3x3<FloatType> matrix3x3() const {

			Quaternion q = getNormalized();
			Matrix3x3<FloatType> m;
			m(0,0) = q.m_Real*q.m_Real + q.m_Imag[0]*q.m_Imag[0] - q.m_Imag[1]*q.m_Imag[1] - q.m_Imag[2]*q.m_Imag[2];
			m(0,1) = (FloatType)2.0 * (q.m_Imag[0]*q.m_Imag[1] - q.m_Real*q.m_Imag[2]);
			m(0,2) = (FloatType)2.0 * (q.m_Imag[0]*q.m_Imag[2] + q.m_Real*q.m_Imag[1]);

			m(1,0) = (FloatType)2.0 * (q.m_Imag[0]*q.m_Imag[1] + q.m_Real*q.m_Imag[2]);
			m(1,1) = q.m_Real*q.m_Real - q.m_Imag[0]*q.m_Imag[0] + q.m_Imag[1]*q.m_Imag[1] - q.m_Imag[2]*q.m_Imag[2];
			m(1,2) = (FloatType)2.0 * (q.m_Imag[1]*q.m_Imag[2] - q.m_Real*q.m_Imag[0]);

			m(2,0) = (FloatType)2.0 * (q.m_Imag[0]*q.m_Imag[2] - q.m_Real*q.m_Imag[1]);
			m(2,1) = (FloatType)2.0 * (q.m_Imag[1]*q.m_Imag[2] + q.m_Real*q.m_Imag[0]);
			m(2,2) = q.m_Real*q.m_Real - q.m_Imag[0]*q.m_Imag[0] - q.m_Imag[1]*q.m_Imag[1] + q.m_Imag[2]*q.m_Imag[2];

			return m;
		}

		//! returns the squared length of the quaternion
		inline FloatType sqrLength() const;
		//! returns the length of the quaternion
		inline FloatType length() const;

		//! return the normalized quaternion as a copy
		inline Quaternion getNormalized() const;
		//! sets the length of the quaternion to one
		inline void normalize();

		//! the absolute-value of a quaternion is its length
		//inline FloatType abs(const Quaternion& q);
		//! the multiplication operator that allows the scalar value to precede the quaternion
		//inline Quaternion operator* (FloatType r, const Quaternion& q);


		//! add two quaternions
		inline Quaternion operator+ ( const Quaternion& b ) const;
		//! add quaternion b to operand
		inline void operator+=( const Quaternion& b );

		inline point3d<FloatType> operator* (const point3d<FloatType>& v) const {
			point3d<FloatType> uv = m_Imag ^ v;
			point3d<FloatType> uuv = m_Imag ^ uv;
			uv *= ((FloatType)2.0 * m_Real);
			uuv *= (FloatType)2.0f;
			return v + uv + uuv;
		}

		//! multiply two quaternions
		inline Quaternion operator* ( const Quaternion& b ) const;
		//! multiply quaternions b to operand
		inline Quaternion operator*=( const Quaternion& b );

		//! scale quaternion with a scalar
		inline Quaternion operator* ( FloatType r ) const;
		//! multiply quaternion with a scalar

		inline void operator*=( FloatType r );
		//! divide quaternion by scalar
		inline Quaternion operator/ ( FloatType r ) const;
		//! divide quaternion by scalar
		inline void operator/=( FloatType r );

		//! returns the scalar-product of the quaternion with the argument
		inline FloatType scalarProd( const Quaternion& b ) const;

		//! return the conjugated quaternion
		inline Quaternion getConjugated() const;

		//! return the quaternion that performs the inverse rotation
		inline Quaternion getInverse() const;
		//! set the quaternion to the one that performs the inverse rotation
		inline void invert();

		//! calculates a spherical linear interpolation between the quaternions
		/*! If t==0 the result is the quaternion from which the method was called.
		    If t==1 the result is the quaternion q2.
		    between the quaternions are weighted
		*/
		Quaternion slerp( const Quaternion& q2, FloatType t ) const;

	private:
		
		void constructFromMatrix( const Matrix3x3<FloatType>& m )
		{
			FloatType m00 = m(0,0);	FloatType m01 = m(0,1);	FloatType m02 = m(0,2);
			FloatType m10 = m(1,0);	FloatType m11 = m(1,1);	FloatType m12 = m(1,2);
			FloatType m20 = m(2,0);	FloatType m21 = m(2,1);	FloatType m22 = m(2,2);

			FloatType tr = m00 + m11 + m22;

			FloatType qw, qx, qy, qz;
			if (tr > 0) { 
				FloatType S = sqrt(tr+(FloatType)1.0) * 2; // S=4*qw 
				qw = (FloatType)0.25 * S;
				qx = (m21 - m12) / S;
				qy = (m02 - m20) / S; 
				qz = (m10 - m01) / S; 
			} else if ((m00 > m11)&(m00 > m22)) { 
				FloatType S = sqrt((FloatType)1.0 + m00 - m11 - m22) * (FloatType)2; // S=4*qx 
				qw = (m21 - m12) / S;
				qx = (FloatType)0.25 * S;
				qy = (m01 + m10) / S; 
				qz = (m02 + m20) / S; 
			} else if (m11 > m22) { 
				FloatType S = sqrt((FloatType)1.0 + m11 - m00 - m22) * (FloatType)2; // S=4*qy
				qw = (m02 - m20) / S;
				qx = (m01 + m10) / S; 
				qy = (FloatType)0.25 * S;
				qz = (m12 + m21) / S; 
			} else { 
				FloatType S = sqrt((FloatType)1.0 + m22 - m00 - m11) * (FloatType)2; // S=4*qz
				qw = (m10 - m01) / S;
				qx = (m02 + m20) / S;
				qy = (m12 + m21) / S;
				qz = (FloatType)0.25 * S;
			}
			m_Real = qw;
			m_Imag = point3d<FloatType>(qx,qy,qz);

			/*
			FloatType r11 = m(0,0);	FloatType r12 = m(0,1);	FloatType r13 = m(0,2);
			FloatType r21 = m(1,0);	FloatType r22 = m(1,1);	FloatType r23 = m(1,2);
			FloatType r31 = m(2,0);	FloatType r32 = m(2,1);	FloatType r33 = m(2,2);

			FloatType q0 = ( r11 + r22 + r33 + 1.0f) / 4.0f;
			FloatType q1 = ( r11 - r22 - r33 + 1.0f) / 4.0f;
			FloatType q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
			FloatType q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
			if(q0 < 0.0f) q0 = 0.0f;
			if(q1 < 0.0f) q1 = 0.0f;
			if(q2 < 0.0f) q2 = 0.0f;
			if(q3 < 0.0f) q3 = 0.0f;
			q0 = sqrt(q0);
			q1 = sqrt(q1);
			q2 = sqrt(q2);
			q3 = sqrt(q3);
			if(q0 >= q1 && q0 >= q2 && q0 >= q3) {
			q0 *= +1.0f;
			q1 *= sgn(r32 - r23);
			q2 *= sgn(r13 - r31);
			q3 *= sgn(r21 - r12);
			} else if(q1 >= q0 && q1 >= q2 && q1 >= q3) {
			q0 *= sgn(r32 - r23);
			q1 *= +1.0f;
			q2 *= sgn(r21 + r12);
			q3 *= sgn(r13 + r31);
			} else if(q2 >= q0 && q2 >= q1 && q2 >= q3) {
			q0 *= sgn(r13 - r31);
			q1 *= sgn(r21 + r12);
			q2 *= +1.0f;
			q3 *= sgn(r32 + r23);
			} else if(q3 >= q0 && q3 >= q1 && q3 >= q2) {
			q0 *= sgn(r21 - r12);
			q1 *= sgn(r31 + r13);
			q2 *= sgn(r32 + r23);
			q3 *= +1.0f;
			} else {
			printf("coding error\n");
			}
			FloatType r = point4d<FloatType>(q0, q1, q2, q3).length();
			q0 /= r;
			q1 /= r;
			q2 /= r;
			q3 /= r;

			re = q3;
			im = point3d<FloatType>(q0,q1,q2);
			*/
		}


		FloatType m_Real;			//! the real part of the quaternion
		point3d<FloatType> m_Imag;	//! the imaginary part of the quaternion

		//! read a quaternion from a stream
		template <class t> friend std::istream& operator>> ( std::istream& s, Quaternion<FloatType>& q );

		static const FloatType EPSILON;
	};

//};	// namespace Math


//namespace Math {

template <class FloatType> const FloatType Quaternion<FloatType>::EPSILON = ( FloatType ) 0.00001;

// ********************************
// INLINE-functions for general use
// ********************************

////! the absolute-value of a quaternion is its length
//template <class FloatType> inline FloatType abs( const Quaternion<FloatType>& q ) { return q.length(); }

//! FloatTypehe multiplication operator that allows the scalar value to preceed the quaternion
template <class FloatType> inline Quaternion<FloatType> operator* ( FloatType r, const Quaternion<FloatType>& q ) { return q * r; }

//! write a quaternion to a stream
template <class FloatType> inline std::ostream& operator<<( std::ostream& s, const Quaternion<FloatType>& q ) { return ( s << q.real() << " " << q.imag() ); }

//! read a quaternion from a stream
template <class FloatType> inline std::istream& operator>>( std::istream& s, Quaternion<FloatType>& q ) { return ( s >> q.m_Real >> q.m_Imag ); }


// ********************************
// INLINE-functions for Quaternion
// ********************************

template <class FloatType> inline FloatType Quaternion<FloatType>::real() const { return m_Real; }

template <class FloatType> inline point3d<FloatType> Quaternion<FloatType>::imag() const { return m_Imag; }

// inline doubleVec Quaternion<FloatType>::LinAlVec() const
// {
//   doubleVec vec(4);
//   vec(0) = re;
//   vec(1) = im[0];
//   vec(2) = im[1];
//   vec(3) = im[2];
//
//   return vec;
// }

template <class FloatType> inline void Quaternion<FloatType>::setReal( const FloatType r ) { m_Real = r; }

template <class FloatType> inline void Quaternion<FloatType>::setImag( const point3d<FloatType>& imag ) { m_Imag = imag; }

template <class FloatType> inline FloatType Quaternion<FloatType>::angleRad() const { return acos( m_Real ); }

template <class FloatType> inline FloatType Quaternion<FloatType>::angleDeg() const { return ( angleRad() * ( 360.0 / M_PI ) ); }

//template <class FloatType> inline Matrix4x4<FloatType> Quaternion<FloatType>::matrix4x4() const {
//	/*
//	Matrix4x4<FloatType> m;
//	
//	float q[ 4 ];
//	q[ 0 ] = m_Imag[ 0 ];
//	q[ 1 ] = m_Imag[ 1 ];
//	q[ 2 ] = m_Imag[ 2 ];
//	q[ 3 ] = m_Real;
//
//	FloatType s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;
//	s = ( FloatType ) 2.0 / ( q[ 0 ] * q[ 0 ] + q[ 1 ] * q[ 1 ] + q[ 2 ] * q[ 2 ] + q[ 3 ] * q[ 3 ] );
//
//	xs = q[ 0 ] * s; ys = q[ 1 ] * s; zs = q[ 2 ] * s;
//	wx = q[ 3 ] * xs; wy = q[ 3 ] * ys; wz = q[ 3 ] * zs;
//	xx = q[ 0 ] * xs; xy = q[ 0 ] * ys; xz = q[ 0 ] * zs;
//	yy = q[ 1 ] * ys; yz = q[ 1 ] * zs; zz = q[ 2 ] * zs;
//
//	m(0,0) = ( FloatType ) 1.0 - ( yy + zz );	m(0,1) = xy + wz;					m(0,2) = xz - wy;					m(0,3) = 0;
//	m(1,0) = xy - wz;					m(1,1) = ( FloatType ) 1.0 - ( xx + zz );	m(1,2) = yz + wx;					m(1,3) = 0;
//	m(2,0) = xz + wy;					m(2,1) = yz - wx;					m(2,2) = ( FloatType ) 1.0 - ( xx + yy );	m(2,3) = 0;
//	m(3,0) = 0;							m(3,1) = 0;							m(3,2) = 0;							m(3,3) = 1;
//	*/
//
//	return Matrix4x4<FloatType>(matrix3x3());
//}


template <class FloatType> inline FloatType Quaternion<FloatType>::sqrLength() const { return m_Real * m_Real + m_Imag[ 0 ] * m_Imag[ 0 ] + m_Imag[ 1 ] * m_Imag[ 1 ] + m_Imag[ 2 ] * m_Imag[ 2 ]; }

template <class FloatType> inline FloatType Quaternion<FloatType>::length() const { return sqrt( sqrLength() ); }

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::getNormalized() const {
	FloatType thisLength = length();
	//if(!isZero(thisLength))
	if ( thisLength > Quaternion<FloatType>::EPSILON )
		return Quaternion( m_Real / thisLength, m_Imag / thisLength );
	else
		return Quaternion( 1, point3d<FloatType>( 0, 0, 0 ) );
	}

template <class FloatType> inline void Quaternion<FloatType>::normalize() {
	FloatType thisLength = length();
	//if(!isZero(thisLength)) {
	if ( thisLength > Quaternion<FloatType>::EPSILON ) {
		m_Real /= thisLength;
		m_Imag /= thisLength;
		} else {
		m_Real = 1;
		m_Imag = point3d<FloatType>( 0, 0, 0 );
		}
	}

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::operator+ ( const Quaternion& b ) const { return Quaternion( b.re + m_Real, b.im + m_Imag ); }

template <class FloatType> inline void Quaternion<FloatType>::operator+=( const Quaternion& b ) { m_Real += b.re; m_Imag += b.im; }

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::operator* ( const Quaternion& b ) const {
	FloatType re2 = ( m_Real * b.real() ) - ( m_Imag | b.imag() );	// | = dot product
	point3d<FloatType> im2 = ( b.imag() * m_Real ) + ( m_Imag * b.real() ) + ( m_Imag ^ b.imag() ); // ^ = cross product
	return Quaternion( re2, im2 );
	}

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::operator*=( const Quaternion& b ) {
	( *this ) = ( *this ) * b;
	return *this;
	}

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::operator* ( FloatType r ) const { return Quaternion( r * m_Real, r * m_Imag ); }

template <class FloatType> inline void Quaternion<FloatType>::operator*=( FloatType r ) { m_Real *= r; m_Imag *= r; }

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::operator/ ( FloatType r ) const {
	assert( !isZero( r ) );
	return Quaternion( m_Real / r, m_Imag / r );
	}

template <class FloatType> inline void Quaternion<FloatType>::operator/=( FloatType r ) {
	assert( !isZero( r ) );
	m_Real /= r; m_Imag /= r;
	}

template <class FloatType> inline FloatType Quaternion<FloatType>::scalarProd( const Quaternion& b ) const { 
	return ( m_Real * b.re + (m_Imag | b.im ) ); 
}

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::getConjugated() const { return Quaternion( m_Real, point3d<FloatType>( -m_Imag[ 0 ], -m_Imag[ 1 ], -m_Imag[ 2 ] ) ); }

template <class FloatType> inline Quaternion<FloatType> Quaternion<FloatType>::getInverse() const {
		FloatType l = length();
		assert(l != (FloatType)0.0);
		//assert( !isZero( l ) );
		return Quaternion( m_Real / l, point3d<FloatType>( -m_Imag[ 0 ], -m_Imag[ 1 ], -m_Imag[ 2 ] ) / l );
	}

template <class FloatType> inline void Quaternion<FloatType>::invert() { *this = getInverse(); }

// ****************************
// IMPLEMENFloatTypeAFloatTypeION of Quaternion
// ****************************



template <class FloatType> point3d<FloatType> Quaternion<FloatType>::axis() const {
	FloatType halfAngle = acos( m_Real );
	FloatType s = sin( halfAngle );
	point3d<FloatType> a;
	if ( isZero( s ) ) {
		a[ 0 ] = 1.0;
		a[ 1 ] = 0.0;
		a[ 2 ] = 0.0;
		} else {
		FloatType c = 1.0 / s;
		a = c * m_Imag;
		}
	return a;
	}

template <class FloatType> Quaternion<FloatType> Quaternion<FloatType>::slerp( const Quaternion<FloatType>& q2, FloatType t ) const {
	const FloatType delta = (FloatType)0.0001;
	Quaternion result;
	FloatType Omega, CosOmega, SinOmega, scale0, scale1;

	// scalar product of the two quaternions is the cosine of the angle between them
	CosOmega = scalarProd( q2 );

	// test if they are exactly opposite
	if ( ( 1.0 + CosOmega ) > delta ) {
		// if they are too close together calculate only lerp
		if ( ( 1.0 - CosOmega ) > delta ) {
			// slerp
			Omega = acos( CosOmega );
			SinOmega = sin( Omega );
			scale0 = sin( ( (FloatType)1.0 - t ) * Omega ) / SinOmega;
			scale1 = sin( t * Omega ) / SinOmega;
			} else {
			// lerp
			scale0 = (FloatType)1.0 - t;
			scale1 = t;
			}
		result.im[ 0 ] = scale0 * m_Imag[ 0 ] + scale1 * q2.im[ 0 ];
		result.im[ 1 ] = scale0 * m_Imag[ 1 ] + scale1 * q2.im[ 1 ];
		result.im[ 2 ] = scale0 * m_Imag[ 2 ] + scale1 * q2.im[ 2 ];
		result.re = scale0 * m_Real + scale1 * q2.re;
		} else {
		// quaternions are opposite
		// calculate orthogonal quaternion and use it to calculate slerp
		result.im[ 0 ] = -q2.im[ 1 ];
		result.im[ 1 ] = q2.im[ 0 ];
		result.im[ 2 ] = -q2.re;
		result.re = q2.im[ 2 ];

		// slerp
		scale0 = sin( ( (FloatType)1.0 - t ) * (FloatType)M_PI / (FloatType)2.0 );
		scale1 = sin( t * (FloatType)M_PI / (FloatType)2.0 );
		result.im[ 0 ] = scale0 * m_Imag[ 0 ] + scale1 * result.im[ 0 ];
		result.im[ 1 ] = scale0 * m_Imag[ 1 ] + scale1 * result.im[ 1 ];
		result.im[ 2 ] = scale0 * m_Imag[ 2 ] + scale1 * result.im[ 2 ];
		result.re = scale0 * m_Real + scale1 * result.re;
		}

	return result;
	}

typedef Quaternion<double> quatd;
typedef Quaternion<float> quatf;

}	// namespace ml

#endif
