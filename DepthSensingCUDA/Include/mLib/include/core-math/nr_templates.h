#ifndef NR_SOLVER_TEMPLATE_H
#define NR_SOLVER_TEMPLATE_H

namespace ml {

//******************************************************************************************

#define NR_TINY 1.0e-20f	// A small number.
#define NR_END 1
//#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau)
#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);a[k][l]=h+s*(g-h*tau)

//******************************************************************************************


/* free a float vector allocated with vector() */
#ifndef NR_FREE_VECTOR
#define NR_FREE_VECTOR
template <class T>
void nr_free_vector(T *v, long nl, long) {
	free((char*) (v+nl-NR_END));
};
#endif


//******************************************************************************************
	

/* Numerical Recipes standard error handler */
#ifndef NR_ERROR
	#define NR_ERROR
	static void nr_error(const char *error_text) {
		std::cerr << "Numerical Recipes run-time error..." << std::endl;
		std::cerr << error_text << std::endl;
		std::cerr << "...now exiting to system..." << std::endl;
		exit(EXIT_FAILURE);
	}
#endif


//******************************************************************************************


/* allocate a float vector with subscript range v[nl..nh] */
#ifndef NR_VECTOR
	#define NR_VECTOR
	template <class T>
	T* nr_vector(long nl, long nh) {
		T *v;
		v=(T *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(T)));
		if (!v) nr_error("allocation failure in vector()");
		return v-nl+NR_END;
	}
#endif


//******************************************************************************************


/*!
  Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition
  of a rowwise permutation of itself. a and n are input. a is output, arranged
  as in equation (2.3.14) above; indx[1..n] is an output vector that records the
  row permutation effected by the partial pivoting: d is output as +- 1 depending
  whether the number of row interchanges was even or odd, respectively. This 
  routine is used in combination with 'lubksb' to solve linear equations or
  invert a matrix.
 */
template<class T>
void ludcmp(T **a, int n, int *indx, T *d) {

	int i,imax,j,k;
	T big,dum,sum,temp;
	T *vv;

	vv=nr_vector<T>(1,n);
	*d=1.0;

  // Loop over rows to get the implicit scaling information
	for (i=1;i<=n;i++) {
		big=0.0;
  
		for (j=1;j<=n;j++) {
			if ((temp=fabs(a[i][j])) > big)
				big=temp;
		}
    
		if (big == 0.0) // No nonzero largest element.
			nr_error("Singular matrix in routine ludcmp");

    // No nonzero largest element.
		vv[i]=1.0f/big; 
	}

  // This is the loop over columns of Crout s method.
	for (j=1;j<=n;j++) {

    // This is equation (2.3.12) except for i = j.
		for (i=1;i<j;i++) {
			sum=a[i][j];

			for (k=1;k<i;k++)
				sum -= a[i][k]*a[k][j];

			a[i][j]=sum;
		} 

    // Initialize for the search for largest pivot element.
		big=0.0;

    // This is i = j of equation (2.3.12) and i = j+1. . .N of equation (2.3.13).
		for (i=j;i<=n;i++) {
			sum=a[i][j];

			for (k=1;k<j;k++)
				sum -= a[i][k]*a[k][j];

			a[i][j]=sum;

      // Is the  gure of merit for the pivot better than the best so far?
			if ( (dum=vv[i]*fabs(sum)) >= big) { 
				big=dum;
				imax=i;
			}
		}

    // Do we need to interchange rows?
		if (j != imax) {
      
      // Yes, do so...
			for (k=1;k<=n;k++) {
				dum=a[imax][k];
				a[imax][k]=a[j][k];
				a[j][k]=dum;
			}

      // ...and change the parity of d.
			*d = -(*d);

      // Also interchange the scale factor.
			vv[imax]=vv[j];
		}


		indx[j]=imax;

		if (a[j][j] == 0.0)
			a[j][j]=NR_TINY;
    
    // If the pivot element is zero the matrix is singular (at least to the precision of the algorithm). For some applications on singular matrices, it is desirable to substitute TINY for zero.
		if (j != n) {
      // Now,  nally, divide by the pivot element.
			dum=1.0f/(a[j][j]);
			for (i=j+1;i<=n;i++)
				a[i][j] *= dum;
		}
	}
	nr_free_vector(vv,1,n);
}


//******************************************************************************************


/*!
  Solves the set of n linear equations A�X = B. Here a[1..n][1..n] is input,
  not as the matrix A but rather as its LU decomposition, determined by the
  routine ludcmp. indx[1..n] is input as the permutation vector returned by
  ludcmp. b[1..n] is input as the right-hand side vector B, and returns with
  the solution vector X. a, n, and indx are not modifed by this routine and
  can be left in place for successive calls with different right-hand sides b.
 */
template <class T>
void lubksb(T **a, int n, int *indx, T b[]) {

	int i,ii=0,ip,j;
	T sum;

  // When ii is set to a positive value, it will become the index of the 
  // first nonvanishing element of b. Wenow do the forward substitution,
  // equation (2.3.6). The only new wrinkle is to unscramble the permutation
  // as we go.
	for (i=1;i<=n;i++) { 
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];

		if (ii)
			for (j=ii;j<=i-1;j++)
				sum -= a[i][j]*b[j];
		else if (sum) //  A nonzero element was encountered, so from now on we will have to do the sums in the loop above.
			ii=i;

		b[i]=sum;
	}

  //  Now we do the backsubstitution, equation (2.3.7).
	for (i=n;i>=1;i--) {
		sum=b[i];
		for (j=i+1;j<=n;j++) 
			sum -= a[i][j]*b[j];

    // Store a component of the solution vector X.
		b[i]=sum/a[i][i];
	}
}


//******************************************************************************************


///*!
//  L�st das LGS a*x = b.
// */
//void solveLU(float **a, int n, float b[]) {
//
//  float d;
//  int* indx = new int[n+1];
//
//  nr_ludcmp(a, n, indx, &d);
//  lubksb(a, n, indx, b);
//}
		

//******************************************************************************************

template<typename T>
inline const T SQR(const T a)
{return a*a;}

template<typename T>
inline const T SIGN(const T &a, const T &b)
{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}
// inline float SIGN(const float &a, const double &b)
// {return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}
// inline float SIGN(const double &a, const float &b)
// {return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}
// template<typename T>
// inline const T MAX(const T &a, const T &b)
// {return b > a ? (b) : (a);}
// template<typename T>
// inline const T MIN(const T &a, const T &b)
// {return b < a ? (b) : (a);}

template<typename T>
T pythag(const T a, const T b)
{
	T absa,absb;
	absa=std::abs(a);
	absb=std::abs(b);
	if (absa > absb)
		return absa*std::sqrt((T)1.0+SQR(absb/absa));
	else
		return (absb == 0.0 ? (T)0.0 : absb*std::sqrt((T)1.0+SQR(absa/absb)));
}

//******************************************************************************************

template<typename T>
bool jacobi(T **a, int n, T d[], T **v, int *nrot) {
	int j,iq,ip,i;
	T tresh,theta,tau,t,sm,s,h,g,c;
	T *b,*z;

	b=nr_vector<T>(1,n);
	z=nr_vector<T>(1,n);
	for (ip=1;ip<=n;ip++) {
		for (iq=1;iq<=n;iq++) v[ip][iq]=(T)0.0;
		v[ip][ip]=(T)1.0;
	}
	for (ip=1;ip<=n;ip++) {
		b[ip]=d[ip]=a[ip][ip];
		z[ip]=(T)0.0;
	}
	*nrot=0;
	for (i=1;i<=50;i++) {
		sm=(T)0.0;
		for (ip=1;ip<=n-1;ip++) {
			for (iq=ip+1;iq<=n;iq++)
				sm += fabs(a[ip][iq]);
			}
			if (sm == 0.0) {
				nr_free_vector(z,1,n);
				nr_free_vector(b,1,n);
				return true;
			}
			if (i < 4)
				tresh=(T)0.2*sm/(n*n);
			else
				tresh=(T)0.0;
			for (ip=1;ip<=n-1;ip++) {
				for (iq=ip+1;iq<=n;iq++) {
					g=(T)100.0*fabs(a[ip][iq]);
					if (i > 4 && (T)(fabs(d[ip])+g) == (T)fabs(d[ip]) && (T)(fabs(d[iq])+g) == (T)fabs(d[iq]))
						a[ip][iq]=(T)0.0;
					else if (fabs(a[ip][iq]) > tresh) {
						h=d[iq]-d[ip];
						if ((T)(fabs(h)+g) == (T)fabs(h))
							t=(a[ip][iq])/h;
						else {
							theta=(T)0.5*h/(a[ip][iq]);
							t=(T)1.0/(fabs(theta)+sqrt((T)1.0+theta*theta));
							if (theta < 0.0)
								t = -t;
						}
						c=(T)1.0/sqrt(1+t*t);
						s=t*c;
						tau=s/((T)1.0+c);
						h=t*a[ip][iq];
						z[ip] -= (T)h;
						z[iq] += (T)h;
						d[ip] -= (T)h;
						d[iq] += (T)h;
						a[ip][iq]=(T)0.0;
						for (j=1;j<=ip-1;j++) {
							ROTATE(a,j,ip,j,iq);
						}
						for (j=ip+1;j<=iq-1;j++) {
							ROTATE(a,ip,j,j,iq);
						}
						for (j=iq+1;j<=n;j++) {
							ROTATE(a,ip,j,iq,j);
						}
						for (j=1;j<=n;j++) {
							ROTATE(v,j,ip,j,iq);
						}
						++(*nrot);
					}
				}
			}
			for (ip=1;ip<=n;ip++) {
				b[ip] += z[ip];
				d[ip]=b[ip];
				z[ip]=(T)0.0;
			}
		}
	return false;
}

//******************************************************************************************

//! Computes the SVD decomposition of a. Returns 'true' on success, otherwise 'false'
template<typename T>
bool svdcmp(T **a, int m, int n, T w[], T **v)
{
	int flag,i,its,j,jj,k,l,nm;
	T anorm,c,f,g,h,s,scale,x,y,z,*rv1;

	rv1=nr_vector<T>(1,n);
	g=scale=anorm=0.0;
	for (i=1;i<=n;i++) {
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i <= m) {
			for (k=i;k<=m;k++) scale += std::abs(a[k][i]);
			if (scale) {
				for (k=i;k<=m;k++) {
					a[k][i] /= scale;
					s += a[k][i]*a[k][i];
				}
				f=a[i][i];
				g = -SIGN(std::sqrt(s),f);
				h=f*g-s;
				a[i][i]=f-g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
					f=s/h;
					for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
				}
				for (k=i;k<=m;k++) a[k][i] *= scale;
			}
		}

		w[i]=scale *g;

		g=s=scale=0.0;
		if (i <= m && i != n) {
			for (k=l;k<=n;k++) scale += std::abs(a[i][k]);
			if (scale) {
				for (k=l;k<=n;k++) {
					a[i][k] /= scale;
					s += a[i][k]*a[i][k];
				}
				f=a[i][l];
				g = -SIGN(std::sqrt(s),f);
				h=f*g-s;
				a[i][l]=f-g;
				for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
				for (j=l;j<=m;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
					for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
				}
				for (k=l;k<=n;k++) a[i][k] *= scale;
			}
		}
		anorm=std::max(anorm,(std::abs(w[i])+std::abs(rv1[i])));
	}

	for (i=n;i>=1;i--) {
		if (i < n) {
			if (g) {
				for (j=l;j<=n;j++)
					v[j][i]=(a[i][j]/a[i][l])/g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
					for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
				}
			}
			for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=std::min(m,n);i>=1;i--) {
		l=i+1;
		g=w[i];
		for (j=l;j<=n;j++) a[i][j]=0.0;
		if (g) {
			g=(T)1.0/g;
			for (j=l;j<=n;j++) {
				for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
				f=(s/a[i][i])*g;
				for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
			}
			for (j=i;j<=m;j++) a[j][i] *= g;
		} else for (j=i;j<=m;j++) a[j][i]=0.0;
		++a[i][i];
	}

	for (k=n;k>=1;k--) {
		for (its=1;its<=30;its++) {
			flag=1;
			for (l=k;l>=1;l--) {
				nm=l-1;
				if ((T)(std::abs(rv1[l])+anorm) == anorm) {
					flag=0;
					break;
				}
				if ((T)(std::abs(w[nm])+anorm) == anorm) break;
			}
			if (flag) {
				c=0.0;
				s=1.0;
				for (i=l;i<=k;i++) {
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((T)(std::abs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=(T)1.0/h;
					c=g*h;
					s = -f*h;
					for (j=1;j<=m;j++) {
						y=a[j][nm];
						z=a[j][i];
						a[j][nm]=y*c+z*s;
						a[j][i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) {
				if (z < 0.0) {
					w[k] = -z;
					for (j=1;j<=n;j++)
						v[j][k] = -v[j][k];
				}
				break;
			}
			if (its == 30) {
				//std::cerr << "NR Error: no convergence in 30 svdcmp iterations" << std::endl;
				return false;
			}

			x=w[l];
			nm=k-1;

			//std::cerr << "nm: " << nm << std::endl;

			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/((T)2.0*h*y);
			g=pythag(f,(T)1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0;
			for (j=l;j<=nm;j++) {

				i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g = g*c-x*s;
				h=y*s;
				y *= c;


				for (jj=1;jj<=n;jj++) {
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z;
				if (z) {
					z=(T)1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=1;jj<=m;jj++) {
					y=a[jj][j];
					z=a[jj][i];
					a[jj][j]=y*c+z*s;
					a[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
	nr_free_vector(rv1,1,n);

	return true;
}


//******************************************************************************************

//******************************************************************************************

//! Computes the SVD decomposition of a. Returns 'true' on success, otherwise 'false'
template<typename T>
bool svdcmpZeroBased(T **a, int m, int n, T w[], T **v)
{
	int flag,i,its,j,jj,k,l,nm;
	T anorm,c,f,g,h,s,scale,x,y,z,*rv1;


	rv1=nr_vector<T>(0,n);
	g=scale=anorm=0.0;
	for (i=0;i<n;i++) {
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i < m) {
			for (k=i;k<m;k++) scale += std::abs(a[k][i]);
			if (scale) {
				for (k=i;k<m;k++) {
					a[k][i] /= scale;
					s += a[k][i]*a[k][i];
				}
				f=a[i][i];
				g = -SIGN(std::sqrt(s),f);
				h=f*g-s;
				a[i][i]=f-g;
				for (j=l;j<n;j++) {
					for (s=0.0,k=i;k<m;k++) s += a[k][i]*a[k][j];
					f=s/h;
					for (k=i;k<m;k++) a[k][j] += f*a[k][i];
				}
				for (k=i;k<m;k++) a[k][i] *= scale;
			}
		}

		w[i]=scale *g;

		g=s=scale=0.0;
		if (i < m && i != (n-1)) {
			for (k=l;k<n;k++) scale += std::abs(a[i][k]);
			if (scale) {
				for (k=l;k<n;k++) {
					a[i][k] /= scale;
					s += a[i][k]*a[i][k];
				}
				f=a[i][l];
				g = -SIGN(std::sqrt(s),f);
				h=f*g-s;
				a[i][l]=f-g;
				for (k=l;k<n;k++) rv1[k]=a[i][k]/h;
				for (j=l;j<m;j++) {
					for (s=0.0,k=l;k<n;k++) s += a[j][k]*a[i][k];
					for (k=l;k<n;k++) a[j][k] += s*rv1[k];
				}
				for (k=l;k<n;k++) a[i][k] *= scale;
			}
		}
		anorm=std::max(anorm,(std::abs(w[i])+std::abs(rv1[i])));
	}

	for (i=(n-1);i>=0;i--) {
		if (i < (n-1)) {
			if (g) {
				for (j=l;j<n;j++)
					v[j][i]=(a[i][j]/a[i][l])/g;
				for (j=l;j<n;j++) {
					for (s=0.0,k=l;k<n;k++) s += a[i][k]*v[k][j];
					for (k=l;k<n;k++) v[k][j] += s*v[k][i];
				}
			}
			for (j=l;j<n;j++) v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=std::min(m,n)-1;i>=0;i--) {
		l=i+1;
		g=w[i];
		for (j=l;j<n;j++) a[i][j]=0.0;
		if (g) {
			g=(T)1.0/g;
			for (j=l;j<n;j++) {
				for (s=0.0,k=l;k<m;k++) s += a[k][i]*a[k][j];
				f=(s/a[i][i])*g;
				for (k=i;k<m;k++) a[k][j] += f*a[k][i];
			}
			for (j=i;j<m;j++) a[j][i] *= g;
		} else for (j=i;j<m;j++) a[j][i]=0.0;
		++a[i][i];
	}

	for (k=(n-1);k>=0;k--) {
		for (its=1;its<=30;its++) {
			flag=1;
			for (l=k;l>=0;l--) {
				nm=l-1;
				if ((T)(std::abs(rv1[l])+anorm) == anorm) {
					flag=0;
					break;
				}
				if ((T)(std::abs(w[nm])+anorm) == anorm) break;
			}
			if (flag) {
				c=0.0;
				s=1.0;
				for (i=l;i<k;i++) {
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if ((T)(std::abs(f)+anorm) == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=(T)1.0/h;
					c=g*h;
					s = -f*h;
					for (j=0;j<m;j++) {
						y=a[j][nm];
						z=a[j][i];
						a[j][nm]=y*c+z*s;
						a[j][i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) {
				if (z < 0.0) {
					w[k] = -z;
					for (j=0;j<n;j++)
						v[j][k] = -v[j][k];
				}
				break;
			}
			if (its == 30) {
				//std::cerr << "NR Error: no convergence in 30 svdcmp iterations" << std::endl;
				return false;
			}


			x=w[l];
			nm=k-1; // <- k-1?

			std::cerr << "nm: " << nm << std::endl;

			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/((T)2.0*h*y);
			g=pythag(f,(T)1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0;
			for (j=l;j<nm;j++) {

			//for (int xx = 0; xx < n; xx++) {
			//	for (int yy = 0; yy < n; yy++) {
			//		std::cerr << a[xx][yy] << " ";
			//	}
			//	std::cerr << std::endl;
			//}
			//std::cerr << std::endl;

				i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=pythag(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g = g*c-x*s;
				h=y*s;
				y *= c;


				for (jj=0;jj<n;jj++) {
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z;
				if (z) {
					z=(T)1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=0;jj<m;jj++) {
					y=a[jj][j];
					z=a[jj][i];
					a[jj][j]=y*c+z*s;
					a[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
	nr_free_vector(rv1,0,n);

	return true;
}


//******************************************************************************************


template<typename T>
void svbksb(T **u, T w[], T **v, int m, int n, T b[], T x[])
{
	int jj,j,i;
	T s,*tmp;

	tmp=nr_vector<T>(1,n);
	for (j=1;j<=n;j++) {
		s=0.0;
		if (w[j]) {
			for (i=1;i<=m;i++)
				s += u[i][j]*b[i];
			s /= w[j];
		}
		tmp[j]=s;
	}
	for (j=1;j<=n;j++) {
		s=0.0;
		for (jj=1;jj<=n;jj++)
			s += v[j][jj]*tmp[jj];
		x[j]=s;
	}
	nr_free_vector(tmp,1,n);
}


#undef NR_END
#undef NR_TINY

} //namespace ml

#endif
