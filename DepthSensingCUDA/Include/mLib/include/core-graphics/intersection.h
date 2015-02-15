
#ifndef CORE_GRAPHICS_INTERSECTION_H_
#define CORE_GRAPHICS_INTERSECTION_H_

namespace ml {

namespace intersection {

    //2D line-line intersection
    template <class T>
    bool intersectLine2Line2(const Line2<T> &lineA, const Line2<T> &lineB, point2d<T> &result)
    {
        //
        // http://www.ahinson.com/algorithms_general/Sections/Geometry/ParametricLineIntersection.pdf
        //

        const point2d<T> d21 = lineA.dir();
        const point2d<T> d43 = lineB.dir();
        const point2d<T> d31 = lineB.p0() - lineA.p0();

        const float d = d43.x * d21.y - d21.x * d43.y;
        if (d == 0.0f)
        {
            return false;
        }
            
        //(bx(cy - ay) + by(ax - cx)) / (dx.by - dy.bx)
        const float tA = (d43.x * d31.y - d31.x * d43.y) / d;
        result = lineA.p0() + tA * lineA.dir();

        //float distA = distSq(lineA, result);
        //float distB = distSq(lineB, result);

        return true;
    }

    //2D line-line intersection
    template <class T>
    bool intersectLineSegment2LineSegment2(const LineSegment2<T> &lineA, const LineSegment2<T> &lineB, point2d<T> &result)
    {
        //
        // http://www.ahinson.com/algorithms_general/Sections/Geometry/ParametricLineIntersection.pdf
        //

        const point2d<T> d21 = lineA.delta();
        const point2d<T> d43 = lineB.delta();
        const point2d<T> d31 = lineB.p0() - lineA.p0();

        const float d = d43.x * d21.y - d21.x * d43.y;
        if (d == 0.0f)
        {
            return false;
        }

        //(bx(cy - ay) + by(ax - cx)) / (dx.by - dy.bx)
        const float tA = (d43.x * d31.y - d31.x * d43.y) / d;
        const float tB = (d21.x * d31.y - d31.x * d21.y) / d;
        if (tA < 0.0f || tA > 1.0f || tB < 0.0f || tB > 1.0f)
            return false;
        
        result = lineA.p0() + tA * lineA.delta();

        //float distA = distSq(lineA, result);
        //float distB = distSq(lineB, result);

        return true;
    }

    template <class FloatType>
    bool intersectRayTriangle(
        const Triangle<FloatType>& triangle, const Ray<FloatType> &r, FloatType& _t, FloatType& _u, FloatType& _v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool intersectOnlyFrontFaces = false)
    {
        return intersectRayTriangle(triangle.vertices[0], triangle.vertices[1], triangle.vertices[2], r, _t, _u, _v, tmin, tmax, intersectOnlyFrontFaces);
    }

	//triangle ray
	template <class FloatType>
	bool intersectRayTriangle(
		const point3d<FloatType>& v0, const point3d<FloatType>& v1, const point3d<FloatType>& v2, const Ray<FloatType> &r, FloatType& _t, FloatType& _u, FloatType& _v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool intersectOnlyFrontFaces = false) 
	{
		const point3d<FloatType> &d = r.direction();
		const point3d<FloatType> &p = r.origin();

		point3d<FloatType> e1 = v1 - v0;
		point3d<FloatType> e2 = v2 - v0;

		if (intersectOnlyFrontFaces) {
			point3d<FloatType> n = e1^e2; n.normalize();
			if ((d | n) > (FloatType)0.0) return false;
		}

		point3d<FloatType> h = d^e2;
		FloatType a = e1 | h;

		//if (a > -0.0000000001 && a < 0.0000000001) return false;
		if (a == (FloatType)0.0 || a == -(FloatType)0.0)	return false;

		FloatType f = (FloatType)1.0/a;
		point3d<FloatType> s = p - v0;
		FloatType u = f * (s | h);

		if (u < (FloatType)0.0 || u > (FloatType)1.0)	return false;

		point3d<FloatType> q = s^e1;
		FloatType v = f * (d | q);

		if (v < (FloatType)0.0 || u + v > (FloatType)1.0)	return false;

		// at this stage we can compute t to find out where the intersection point is on the line
		FloatType t = f * (e2 | q);

		//if (t > 0.0000000001 && t < r.t) {
		//if (t < _t) {
		if (t <= tmax && t >= tmin) {
			_t = t;
			_u = u;
			_v = v;
			return true;
		} else {
			return false;
		}

	}







	// sort so that a<=b 
	template<class FloatType>
	void SORT(FloatType& a, FloatType& b) {
		if (a > b) {
			std::swap(a,b);
		}
	}

	// this edge to edge test is based on Franlin Antonio's gem:
    //"Faster Line Segment Intersection", in Graphics Gems III,
    //pp. 199-202
	template<class FloatType> 
	bool EDGE_EDGE_TEST(
		const point3d<FloatType>& V0, 
		const point3d<FloatType>& U0, 
		const point3d<FloatType>& U1,
		unsigned short i0, unsigned short i1,
		point2d<FloatType>& a,
		point2d<FloatType>& b, 
		point2d<FloatType>& c,
		FloatType& d,
		FloatType& e,
		FloatType& f) 
	{

		b.x = U0[i0] - U1[i0];
		b.y = U0[i1] - U1[i1];
		c.x = V0[i0] - U0[i0];
		c.y = V0[i1] - U0[i1];

		f = a.y*b.x - a.x*b.y;
		d = b.y*c.x - b.x*c.y;

		if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f)) 	{                                                  
			e=a.x*c.y-a.y*c.x;                                   
			if (f>0)	{                                                
				if(e>=0 && e<=f) {return true;}            
			} else {                                                
				if(e<=0 && e>=f) {return true;}                       
			}                                                
		}
		return false;
	}
	

	template<class FloatType> 
	bool EDGE_AGAINST_TRI_EDGES(
		const point3d<FloatType>& V0,
		const point3d<FloatType>& V1,
		const point3d<FloatType>& U0,
		const point3d<FloatType>& U1,
		const point3d<FloatType>& U2,
		unsigned short i0, unsigned short i1)
	{
		point2d<FloatType> a, b, c;
		FloatType e,d,f;          
		a.x = V1[i0]-V0[i0];                           
		a.y = V1[i1]-V0[i1];                           
		//test edge U0,U1 against V0,V1         
		if (EDGE_EDGE_TEST(V0,U0,U1,i0,i1,a,b,c,d,e,f)) return true;                
		//test edge U1,U2 against V0,V1        
		if (EDGE_EDGE_TEST(V0,U1,U2,i0,i1,a,b,c,d,e,f)) return true;                 
		//test edge U2,U1 against V0,V1    
		if (EDGE_EDGE_TEST(V0,U2,U0,i0,i1,a,b,c,d,e,f)) return true;

		return false;
	}


	template<class FloatType> 
	bool POINT_IN_TRI(
		const point3d<FloatType>& V0,
		const point3d<FloatType>& U0,
		const point3d<FloatType>& U1,
		const point3d<FloatType>& U2,
		unsigned short i0, unsigned short i1)          
	{                                          
		FloatType a,b,c,d0,d1,d2;                    
		//is T1 completely inside T2?    
		//check if V0 is inside tri(U0,U1,U2)
		a=U1[i1]-U0[i1];                         
		b=-(U1[i0]-U0[i0]);                      
		c=-a*U0[i0]-b*U0[i1];                    
		d0=a*V0[i0]+b*V0[i1]+c;                  

		a=U2[i1]-U1[i1];                         
		b=-(U2[i0]-U1[i0]);                      
		c=-a*U1[i0]-b*U1[i1];                    
		d1=a*V0[i0]+b*V0[i1]+c;                  

		a=U0[i1]-U2[i1];                         
		b=-(U0[i0]-U2[i0]);                      
		c=-a*U2[i0]-b*U2[i1];                    
		d2=a*V0[i0]+b*V0[i1]+c;                  
		if (d0*d1 > (FloatType)0.0) {                                        
			if (d0*d2 > (FloatType)0.0) return true;               
		} 
		return false;
	}


	template<class FloatType>
	bool coplanar_tri_tri(
		const point3d<FloatType>& N,
		const point3d<FloatType>& V0,
		const point3d<FloatType>& V1,
		const point3d<FloatType>& V2,
		const point3d<FloatType>& U0,
		const point3d<FloatType>& U1,
		const point3d<FloatType>& U2)
	{
		point3d<FloatType> A;
		short i0,i1;
		// first project onto an axis-aligned plane, that maximizes the area 
		// of the triangles, compute indices: i0,i1. 
		A[0] = math::abs(N[0]);
		A[1] = math::abs(N[1]);
		A[2] = math::abs(N[2]);
		if(A[0]>A[1])	{
			if(A[0]>A[2])	{
				i0=1;      // A[0] is greatest
				i1=2;
			} else	{
				i0=0;      // A[2] is greatest
				i1=1;
			}
		} 
		else   // A[0]<=A[1] 
		{
			if(A[2]>A[1]) {
				i0=0;      // A[2] is greatest 
				i1=1;
			} else {
				i0=0;      // A[1] is greatest 
				i1=2;
			}
		}

		// test all edges of triangle 1 against the edges of triangle 2 
		if (EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2,i0,i1))	return true;
		if (EDGE_AGAINST_TRI_EDGES(V1,V2,U0,U1,U2,i0,i1))	return true;
		if (EDGE_AGAINST_TRI_EDGES(V2,V0,U0,U1,U2,i0,i1))	return true;

		// finally, test if tri1 is totally contained in tri2 or vice versa 
		if (POINT_IN_TRI(V0,U0,U1,U2,i0,i1)) return true;
		if (POINT_IN_TRI(U0,V0,V1,V2,i0,i1)) return true;

		return false;
	}


	//returns true if coplanar
	template<class FloatType>
	bool NEWCOMPUTE_INTERVALS(
		const point3d<FloatType>& v0,
		const point3d<FloatType>& v1,
		const point3d<FloatType>& v2,
		const point3d<FloatType>& u0,
		const point3d<FloatType>& u1,
		const point3d<FloatType>& u2,
		const point3d<FloatType>& n1,
		FloatType& VV0,FloatType& VV1,FloatType& VV2, 
		FloatType& D0, FloatType& D1, FloatType& D2, 
		FloatType& D0D1, FloatType& D0D2,
		FloatType& A, FloatType& B, FloatType& C, FloatType& X0, FloatType& X1) 
	{
		if(D0D1>(FloatType)0.0)
		{
			// here we know that D0D2<=0.0
			// that is D0, D1 are on the same side, D2 on the other or on the plane
			A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1;
		}	
		else if(D0D2>(FloatType)0.0)
		{ 
			// here we know that d0d1<=0.0
			A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2;
		}
		else if(D1*D2>(FloatType)0.0 || D0!=(FloatType)0.0)
		{
			// here we know that d0d1<=0.0 or that D0!=0.0
			A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; 
		} 
		else if(D1!=(FloatType)0.0)
		{
			A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2;
		}
		else if(D2!=(FloatType)0.0)
		{
			A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1;
		}
		else
		{
			//triangles are co-planar
			return true;
		}
		return false;
	}

	template<class FloatType>
	bool intersectTriangleTriangle(
		const point3d<FloatType> &v0,
		const point3d<FloatType> &v1,
		const point3d<FloatType> &v2,
		const point3d<FloatType> &u0,
		const point3d<FloatType> &u1,
		const point3d<FloatType> &u2) 
	{
		const bool USE_EPSILON_TEST = true;
		const FloatType EPSILON = (FloatType)0.000001;

		//compute plane equation of triangle(V0,V1,V2)
		point3d<FloatType> e1 = v1 - v0;
		point3d<FloatType> e2 = v2 - v0;
		point3d<FloatType> n1 = e1 ^ e2;
		FloatType d1 = - n1 | v0;
		//plane equation 1: N1.X+d1=0 */

		//put U0,U1,U2 into plane equation 1 to compute signed distances to the plane
		FloatType du0 = (n1 | u0)+d1;
		FloatType du1 = (n1 | u1)+d1;
		FloatType du2 = (n1 | u2)+d1;

		//co-planarity robustness check
//#if USE_EPSILON_TEST==TRUE
		if (USE_EPSILON_TEST) {
			if(math::abs(du0)<EPSILON) du0=(FloatType)0.0;
			if(math::abs(du1)<EPSILON) du1=(FloatType)0.0;
			if(math::abs(du2)<EPSILON) du2=(FloatType)0.0;
		}
//#endif
		FloatType du0du1 = du0*du1;
		FloatType du0du2 = du0*du2;

		if (du0du1 > (FloatType)0.0 && du0du2 > (FloatType)0.0) //same sign on all of them + not equal 0 ? 
			return false;										//no intersection occurs

		// compute plane of triangle (U0,U1,U2) 
		e1 = u1 - u0;
		e2 = u2 - u0;
		point3d<FloatType> n2 = e1 ^ e2;
		FloatType d2 = -(n2 | u0);
		// plane equation 2: N2.X+d2=0 

		// put V0,V1,V2 into plane equation 2 
		FloatType dv0 = (n2 | v0)+d2;
		FloatType dv1 = (n2 | v1)+d2;
		FloatType dv2 = (n2 | v2)+d2;

//#if USE_EPSILON_TEST==TRUE
		if (USE_EPSILON_TEST) {
			if(math::abs(dv0)<EPSILON) dv0 = (FloatType)0.0;
			if(math::abs(dv1)<EPSILON) dv1 = (FloatType)0.0;
			if(math::abs(dv2)<EPSILON) dv2 = (FloatType)0.0;
		}
//#endif

		FloatType dv0dv1=dv0*dv1;
		FloatType dv0dv2=dv0*dv2;

		if(dv0dv1>(FloatType)0.0 && dv0dv2>(FloatType)0.0) //same sign on all of them + not equal 0 ?
			return false;                    // no intersection occurs

		// compute direction of intersection line
		point3d<FloatType> D = (n1^n2);

		// compute and index to the largest component of D
		FloatType max = (FloatType)math::abs(D[0]);
		unsigned int index=0;
		FloatType bb=(FloatType)math::abs(D[1]);
		FloatType cc=(FloatType)math::abs(D[2]);
		if(bb>max) max=bb,index=1;
		if(cc>max) max=cc,index=2;

		// this is the simplified projection onto L
		FloatType vp0=v0[index];
		FloatType vp1=v1[index];
		FloatType vp2=v2[index];

		FloatType up0=u0[index];
		FloatType up1=u1[index];
		FloatType up2=u2[index];

		// compute interval for triangle 1
		FloatType a,b,c,x0,x1;
		if (NEWCOMPUTE_INTERVALS(u0,u1,u2,v0,v1,v2,n1,vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,a,b,c,x0,x1)) {
			return (coplanar_tri_tri(n1,v0,v1,v2,u0,u1,u2));	//triangles are a co-planar
		}

		// compute interval for triangle 2 
		FloatType d,e,f,y0,y1;
		if (NEWCOMPUTE_INTERVALS(u0,u1,u2,v0,v1,v2,n1,up0,up1,up2,du0,du1,du2,du0du1,du0du2,d,e,f,y0,y1)) {
			return (coplanar_tri_tri(n1,v0,v1,v2,u0,u1,u2));	//triangles are a co-planar
		}

		FloatType xx,yy,xxyy,tmp;
		xx=x0*x1;
		yy=y0*y1;
		xxyy=xx*yy;

		point2d<FloatType> isect1, isect2;

		tmp=a*xxyy;
		isect1[0]=tmp+b*x1*yy;
		isect1[1]=tmp+c*x0*yy;

		tmp=d*xxyy;
		isect2[0]=tmp+e*xx*y1;
		isect2[1]=tmp+f*xx*y0;

		SORT(isect1[0],isect1[1]);
		SORT(isect2[0],isect2[1]);

		if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return false;
		return true;

	}



//
////////////////////////////////////////////////////////
//// Triangle ABBB: http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
////////////////////////////////////////////////////////
//	template<class FloatType>
//	void FINDMINMAX(FloatType x0, FloatType x1, FloatType x2, FloatType& min, FloatType& max) {
//		min = max = x0;
//		if(x1<min) min=x1;
//		if(x1>max) max=x1;
//		if(x2<min) min=x2;
//		if(x2>max) max=x2;
//	}
//
//
//	//I think this test is somehow broken :) - it's not being used at this point...
//	template<class FloatType>
//	bool intersectTriangleABBB2(
//		const point3d<FloatType> &bbBoxMin,
//		const point3d<FloatType> &bbBoxMax,
//		const point3d<FloatType> &t0,
//		const point3d<FloatType> &t1,
//		const point3d<FloatType> &t2) 
//	{
//		const point3d<FloatType> boxcenter = (FloatType)0.5*(bbBoxMax + bbBoxMin);
//		const point3d<FloatType> boxhalfsize = (FloatType)0.5*(bbBoxMax - bbBoxMin);
//
//		//use separating axis theorem to test overlap between triangle and box
//		//need to test for overlap in these directions:
//		//1) the {x,y,z}-directions (actually, since we use the AABB of the triangle 
//		//we do not even need to test these) 
//		//2) normal of the triangle 
//		//3) crossproduct(edge from tri, {x,y,z}-directin)
//		//this gives 3x3=9 more tests
//
//		//   float axis[3];
//		FloatType min,max,p0,p1,p2,rad,fex,fey,fez;		// -NJMP- "d" local variable removed
//				
//		//This is the fastest branch on Sun 
//		//move everything so that the boxcenter is in (0,0,0)
//		point3d<FloatType> v0 = t0 - boxcenter;
//		point3d<FloatType> v1 = t1 - boxcenter;
//		point3d<FloatType> v2 = t2 - boxcenter;
//
//		//compute triangle edges
//		point3d<FloatType> e0 = v1 - v0;	//tri edge 0
//		point3d<FloatType> e1 = v2 - v1;	//tri edge 1
//		point3d<FloatType> e2 = v0 - v2;	//tri edge 2
//		
//		//Bullet 3:
//		//test the 9 tests first (this was faster)
//
//		fex = math::abs(e0.x);
//		fey = math::abs(e0.y);
//		fez = math::abs(e0.z);
//
//		//AXISTEST_X01(e0.z, e0.y, fez, fey);
//		{
//			FloatType a = e0.z;
//			FloatType b = e0.y;
//			FloatType fa = fez;
//			FloatType fb = fey;
//			p0 = a*v0.y - b*v0.z;	
//			p2 = a*v2.y - b*v2.z;
//			if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} 
//			rad = fa * boxhalfsize.y + fb * boxhalfsize.z;   
//			if(min>rad || max<-rad) return false;
//		}		
//		//AXISTEST_Y02(e0.z, e0.x, fez, fex);
//		{
//			FloatType a = e0.z;
//			FloatType b = e0.x;
//			FloatType fa = fez;
//			FloatType fb = fex;
//			p0 = -a*v0.x + b*v0.z;
//			p2 = -a*v2.x + b*v2.z;
//			if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
//			if(min>rad || max<-rad) return false;
//		}
//		//AXISTEST_Z12(e0.y, e0.x, fey, fex);
//		{
//			FloatType a = e0.y;
//			FloatType b = e0.x;
//			FloatType fa = fey;
//			FloatType fb = fex;
//			p1 = a*v1.x - b*v1.y;
//			p2 = a*v2.x - b*v2.y;
//			if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
//			if(min>rad || max<-rad) return false;
//		}
//
//		fex = math::abs(e1.x);
//		fey = math::abs(e1.y);
//		fez = math::abs(e1.x);
//
//		//AXISTEST_X01(e1[Z], e1[Y], fez, fey);
//		{
//			FloatType a = e1.z;
//			FloatType b = e1.y;
//			FloatType fa = fez;
//			FloatType fb = fey;
//			p0 = a*v0.y - b*v0.z;	
//			p2 = a*v2.y - b*v2.z;
//			if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;} 
//			rad = fa * boxhalfsize.y + fb * boxhalfsize.z;   
//			if(min>rad || max<-rad) return false;
//		}
//		//AXISTEST_Y02(e1[Z], e1[X], fez, fex);
//		{
//			FloatType a = e1.z;
//			FloatType b = e1.x;
//			FloatType fa = fez;
//			FloatType fb = fex;
//			p0 = -a*v0.x + b*v0.z;
//			p2 = -a*v2.x + b*v2.z;
//			if(p0<p2) {min=p0; max=p2;} else {min=p2; max=p0;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
//			if(min>rad || max<-rad) return false;
//		}
//		//AXISTEST_Z0(e1[Y], e1[X], fey, fex);
//		{
//			FloatType a = e1.y;
//			FloatType b = e1.x;
//			FloatType fa = fey;
//			FloatType fb = fex;
//			p0 = a*v0.x - b*v0.y;	
//			p1 = a*v1.x - b*v1.y;
//			if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
//			if(min>rad || max<-rad) return false;
//		}
//
//		fex = math::abs(e2.x);
//		fey = math::abs(e2.y);
//		fez = math::abs(e2.z);
//
//		//AXISTEST_X2(e2[Z], e2[Y], fez, fey);
//		{
//			FloatType a = e2.z;
//			FloatType b = e2.y;
//			FloatType fa = fez;
//			FloatType fb = fey;
//			p0 = a*v0.y - b*v0.z;
//			p1 = a*v1.y - b*v1.z;
//			if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}
//			rad = fa * boxhalfsize.y + fb * boxhalfsize.z;
//			if(min>rad || max<-rad) return false;
//		}
//		//AXISTEST_Y1(e2[Z], e2[X], fez, fex);
//		{
//			FloatType a = e2.z;
//			FloatType b = e2.x;
//			FloatType fa = fez;
//			FloatType fb = fex;
//			p0 = -a*v0.x + b*v0.x;
//			p1 = -a*v1.x + b*v1.x;
//			if(p0<p1) {min=p0; max=p1;} else {min=p1; max=p0;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.z;
//			if(min>rad || max<-rad) return false;
//		}
//		//AXISTEST_Z12(e2[Y], e2[X], fey, fex);
//		{
//			FloatType a = e2.y;
//			FloatType b = e2.x;
//			FloatType fa = fey;
//			FloatType fb = fex;
//			p1 = a*v1.x - b*v1.y;
//			p2 = a*v2.x - b*v2.y;
//			if(p2<p1) {min=p2; max=p1;} else {min=p1; max=p2;}
//			rad = fa * boxhalfsize.x + fb * boxhalfsize.y;
//			if(min>rad || max<-rad) return false;
//		}
//		
//		//Bullet 1:
//		//first test overlap in the {x,y,z}-directions
//		//find min, max of the triangle each direction, and test for overlap in
//		//that direction -- this is equivalent to testing a minimal AABB around
//		//the triangle against the AABB
//		
//		//test in X-direction
//		FINDMINMAX(v0.x,v1.x,v2.x,min,max);
//		if(min>boxhalfsize.x || max<-boxhalfsize.x) return false;
//
//		//test in Y-direction
//		FINDMINMAX(v0.y,v1.y,v2.y,min,max);
//		if(min>boxhalfsize.y || max<-boxhalfsize.y) return false;
//		
//		//test in Z-direction
//		FINDMINMAX(v0.z,v1.z,v2.z,min,max);
//		if(min>boxhalfsize.z || max<-boxhalfsize.z) return false;
//		
//		//Bullet 2:
//		//test if the box intersects the plane of the triangle 
//		//compute plane equation of triangle: normal*x+d=0
//
//		point3d<FloatType> normal = e0 ^ e1;
//
//		// -NJMP- (line removed here)
//		//if(!planeBoxOverlap(normal,v0,boxhalfsize)) return false;	// -NJMP-
//		{
//			point3d<FloatType> vert = v0;
//			point3d<FloatType> maxbox = boxhalfsize;
//			//int planeBoxOverlap(float normal[3], float vert[3], float maxbox[3])	// -NJMP-
//
//			point3d<FloatType> vmin,vmax;
//
//			for (unsigned int q = 0; q<=2; q++) {
//				FloatType v = vert[q];					// -NJMP-
//				if (normal[q]>0.0f) {
//					vmin[q]=-maxbox[q] - v;	// -NJMP-
//					vmax[q]= maxbox[q] - v;	// -NJMP-
//				} else {
//					vmin[q]= maxbox[q] - v;	// -NJMP-
//					vmax[q]=-maxbox[q] - v;	// -NJMP-
//				}
//			}
//
//			if ((normal | vmin) > (FloatType)0.0) return false;	// -NJMP-
//			if ((normal | vmax) >= (FloatType)0.0) return true;	// -NJMP-
//
//			return false;
//		}		
//
//		return true;  //box and triangle overlaps 
//	}



	//! returns 1 if intersects; 0 if front; -1 if back
	template<class FloatType>
	int intersectBoxPlane(
		const point3d<FloatType>& bbBoxMin,
		const point3d<FloatType>& bbBoxMax,
		const Plane<FloatType>& plane) 
	{
		FloatType BOX_PLANE_EPSILON = (FloatType)0.00001;

		point3d<FloatType> center = (FloatType)0.5 * (bbBoxMin + bbBoxMax);
		point3d<FloatType> extent = bbBoxMax - center;
		FloatType _fOrigin =  plane.getNormal() | center;
		FloatType _fMaximumExtent = extent | math::abs(plane.getNormal());
		FloatType _fmin = _fOrigin - _fMaximumExtent;
		FloatType _fmax = _fOrigin + _fMaximumExtent;

		if (plane.getDistance() > _fmax + BOX_PLANE_EPSILON) {
			return -1;
			//return G_BACK_PLANE; // 0
		}

		if (plane.getDistance() + BOX_PLANE_EPSILON >=_fmin) {
			return 1;
			//return G_COLLIDE_PLANE; //1
		}
		return 0;
		//return G_FRONT_PLANE;//2
	}


#define TEST_CROSS_EDGE_BOX_MCR(edge,absolute_edge,pointa,pointb,_extend,i_dir_0,i_dir_1,i_comp_0,i_comp_1)\
{\
	const FloatType dir0 = -edge[i_dir_0];\
	const FloatType dir1 = edge[i_dir_1];\
	FloatType pmin = pointa[i_comp_0]*dir0 + pointa[i_comp_1]*dir1;\
	FloatType pmax = pointb[i_comp_0]*dir0 + pointb[i_comp_1]*dir1;\
	if(pmin>pmax)\
	{\
	std::swap(pmin,pmax); \
	}\
	const FloatType abs_dir0 = absolute_edge[i_dir_0];\
	const FloatType abs_dir1 = absolute_edge[i_dir_1];\
	const FloatType rad = _extend[i_comp_0] * abs_dir0 + _extend[i_comp_1] * abs_dir1;\
	if(pmin>rad || -rad>pmax) return false;\
}\


#define TEST_CROSS_EDGE_BOX_X_AXIS_MCR(edge,absolute_edge,pointa,pointb,_extend)\
{\
	TEST_CROSS_EDGE_BOX_MCR(edge,absolute_edge,pointa,pointb,_extend,2,1,1,2);\
}\

#define TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(edge,absolute_edge,pointa,pointb,_extend)\
{\
	TEST_CROSS_EDGE_BOX_MCR(edge,absolute_edge,pointa,pointb,_extend,0,2,2,0);\
}\

#define TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(edge,absolute_edge,pointa,pointb,_extend)\
{\
	TEST_CROSS_EDGE_BOX_MCR(edge,absolute_edge,pointa,pointb,_extend,1,0,0,1);\
}\


	////////////from BULLET///////////////
	template<class FloatType>
	bool intersectTriangleAABB(
		const point3d<FloatType> &bbBoxMin,
		const point3d<FloatType> &bbBoxMax,
		const point3d<FloatType> &p0,
		const point3d<FloatType> &p1,
		const point3d<FloatType> &p2) 
	{
		//TODO
		//if(!collide_plane(triangle_plane)) return false;
		if (intersectBoxPlane(bbBoxMin, bbBoxMax, Plane<FloatType>(p0,p1,p2)) != 1) return false;

		point3d<FloatType> center = (FloatType)0.5 * (bbBoxMin + bbBoxMax);
		point3d<FloatType> extent = bbBoxMax - center;

		const point3d<FloatType> v1(p0 - center);
		const point3d<FloatType> v2(p1 - center);
		const point3d<FloatType> v3(p2 - center);

		//First axis
		point3d<FloatType> diff(v2 - v1);
		point3d<FloatType> abs_diff = math::abs(diff);
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff,abs_diff,v1,v3,extent);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff,abs_diff,v1,v3,extent);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff,abs_diff,v1,v3,extent);


		diff = v3 - v2;
		abs_diff = math::abs(diff);
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff,abs_diff,v2,v1,extent);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff,abs_diff,v2,v1,extent);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff,abs_diff,v2,v1,extent);

		diff = v1 - v3;
		abs_diff = math::abs(diff);
		//Test With X axis
		TEST_CROSS_EDGE_BOX_X_AXIS_MCR(diff,abs_diff,v3,v2,extent);
		//Test With Y axis
		TEST_CROSS_EDGE_BOX_Y_AXIS_MCR(diff,abs_diff,v3,v2,extent);
		//Test With Z axis
		TEST_CROSS_EDGE_BOX_Z_AXIS_MCR(diff,abs_diff,v3,v2,extent);

		return true;

	}

	template<class FloatType> 
	FloatType halfEdgeTest(const point2d<FloatType>& p1, const point2d<FloatType>& p2, const point2d<FloatType>& p3) {
		return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
	}

	//2d test whether the point inside of the triangle (eps makes the triangle bigger)
	template<class FloatType>
	bool intersectTrianglePoint(
		const point2d<FloatType> &v1,
		const point2d<FloatType> &v2,
		const point2d<FloatType> &v3,
		const point2d<FloatType> &pt) 
	{
		bool b1, b2, b3;

		b1 = halfEdgeTest(pt, v1, v2) < (FloatType)0.0;
		b2 = halfEdgeTest(pt, v2, v3) < (FloatType)0.0;
		b3 = halfEdgeTest(pt, v3, v1) < (FloatType)0.0;

		return ((b1 == b2) && (b2 == b3));
	}

	template<class FloatType>
	bool intersectOBBOBB(
		const point3d<FloatType>& anchor0, const point3d<FloatType>* axesScaled0,
		const point3d<FloatType>& anchor1, const point3d<FloatType>* axesScaled1) 
	{
		struct Result
		{
			// The 'epsilon' value must be nonnegative.
			Result(FloatType inEpsilon = (FloatType)0) {
				if (inEpsilon >= (FloatType)0)	{
					epsilon = inEpsilon;
				} else {
					epsilon = (FloatType)0;
				}
			}

			bool intersect;
			FloatType epsilon;
			int separating[2];
		};


		Result result;

		// Convenience variables.
		//point3d<FloatType> const& C0 = box0.center;
		//point3d<FloatType> const* A0 = &box0.axis[0];
		//point3d<FloatType> const& E0 = box0.extent;
		//point3d<FloatType> const& C1 = box1.center;
		//point3d<FloatType> const* A1 = &box1.axis[0];
		//point3d<FloatType> const& E1 = box1.extent;

		std::array<point3d<FloatType>, 3> A0;	
		A0[0] = axesScaled0[0].getNormalized();	
		A0[1] = axesScaled0[1].getNormalized();	
		A0[2] = axesScaled0[2].getNormalized();
		point3d<FloatType> E0 = point3d<FloatType>(axesScaled0[0].length(), axesScaled0[1].length(), axesScaled0[2].length()) / 2;
		point3d<FloatType> C0 = anchor0 + (axesScaled0[0] + axesScaled0[1] + axesScaled0[2])/2;

		std::array<point3d<FloatType>, 3> A1;	
		A1[0] = axesScaled1[0].getNormalized();	
		A1[1] = axesScaled1[1].getNormalized();	
		A1[2] = axesScaled1[2].getNormalized();
		point3d<FloatType> E1 = point3d<FloatType>(axesScaled1[0].length(), axesScaled1[1].length(), axesScaled1[2].length()) / 2;
		point3d<FloatType> C1 = anchor1 + (axesScaled1[0] + axesScaled1[1] + axesScaled1[2]) / 2;;

		const FloatType cutoff = (FloatType)1 - result.epsilon;
		bool existsParallelPair = false;

		// Compute difference of box centers.
		point3d<FloatType> D = C1 - C0;

		FloatType dot01[3][3];       // dot01[i][j] = Dot(A0[i],A1[j]) = A1[j][i]
		FloatType absDot01[3][3];    // |dot01[i][j]|
		FloatType dotDA0[3];         // Dot(D, A0[i])
		FloatType r0, r1, r;         // interval radii and distance between centers
		FloatType r01;               // r0 + r1

		// Test for separation on the axis C0 + t*A0[0].
		for (int i = 0; i < 3; ++i)
		{
			dot01[0][i] = point3d<FloatType>::dot(A0[0], A1[i]);
			absDot01[0][i] = std::abs(dot01[0][i]);
			if (absDot01[0][i] > cutoff)
			{
				existsParallelPair = true;
			}
		}
		dotDA0[0] = point3d<FloatType>::dot(D, A0[0]);
		r = std::abs(dotDA0[0]);
		r1 = E1[0] * absDot01[0][0] + E1[1] * absDot01[0][1] + E1[2] * absDot01[0][2];
		r01 = E0[0] + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 0;
			result.separating[1] = -1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[1].
		for (int i = 0; i < 3; ++i)
		{
			dot01[1][i] = point3d<FloatType>::dot(A0[1], A1[i]);
			absDot01[1][i] = std::abs(dot01[1][i]);
			if (absDot01[1][i] > cutoff)
			{
				existsParallelPair = true;
			}
		}
		dotDA0[1] = point3d<FloatType>::dot(D, A0[1]);
		r = std::abs(dotDA0[1]);
		r1 = E1[0] * absDot01[1][0] + E1[1] * absDot01[1][1] + E1[2] * absDot01[1][2];
		r01 = E0[1] + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 1;
			result.separating[1] = -1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[2].
		for (int i = 0; i < 3; ++i)
		{
			dot01[2][i] = point3d<FloatType>::dot(A0[2], A1[i]);
			absDot01[2][i] = std::abs(dot01[2][i]);
			if (absDot01[2][i] > cutoff)
			{
				existsParallelPair = true;
			}
		}
		dotDA0[2] = point3d<FloatType>::dot(D, A0[2]);
		r = std::abs(dotDA0[2]);
		r1 = E1[0] * absDot01[2][0] + E1[1] * absDot01[2][1] + E1[2] * absDot01[2][2];
		r01 = E0[2] + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 2;
			result.separating[1] = -1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A1[0].
		r = std::abs(point3d<FloatType>::dot(D, A1[0]));
		r0 = E0[0] * absDot01[0][0] + E0[1] * absDot01[1][0] + E0[2] * absDot01[2][0];
		r01 = r0 + E1[0];
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = -1;
			result.separating[1] = 0;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A1[1].
		r = std::abs(point3d<FloatType>::dot(D, A1[1]));
		r0 = E0[0] * absDot01[0][1] + E0[1] * absDot01[1][1] + E0[2] * absDot01[2][1];
		r01 = r0 + E1[1];
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = -1;
			result.separating[1] = 1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A1[2].
		r = std::abs(point3d<FloatType>::dot(D, A1[2]));
		r0 = E0[0] * absDot01[0][2] + E0[1] * absDot01[1][2] + E0[2] * absDot01[2][2];
		r01 = r0 + E1[2];
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = -1;
			result.separating[1] = 2;
			return result.intersect;
		}

		// At least one pair of box axes was parallel, so the separation is
		// effectively in 2D.  The edge-edge axes do not need to be tested.
		if (existsParallelPair)
		{
			return true;
		}

		// Test for separation on the axis C0 + t*A0[0]xA1[0].
		r = std::abs(dotDA0[2] * dot01[1][0] - dotDA0[1] * dot01[2][0]);
		r0 = E0[1] * absDot01[2][0] + E0[2] * absDot01[1][0];
		r1 = E1[1] * absDot01[0][2] + E1[2] * absDot01[0][1];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 0;
			result.separating[1] = 0;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[0]xA1[1].
		r = std::abs(dotDA0[2] * dot01[1][1] - dotDA0[1] * dot01[2][1]);
		r0 = E0[1] * absDot01[2][1] + E0[2] * absDot01[1][1];
		r1 = E1[0] * absDot01[0][2] + E1[2] * absDot01[0][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 0;
			result.separating[1] = 1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[0]xA1[2].
		r = std::abs(dotDA0[2] * dot01[1][2] - dotDA0[1] * dot01[2][2]);
		r0 = E0[1] * absDot01[2][2] + E0[2] * absDot01[1][2];
		r1 = E1[0] * absDot01[0][1] + E1[1] * absDot01[0][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 0;
			result.separating[1] = 2;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[1]xA1[0].
		r = std::abs(dotDA0[0] * dot01[2][0] - dotDA0[2] * dot01[0][0]);
		r0 = E0[0] * absDot01[2][0] + E0[2] * absDot01[0][0];
		r1 = E1[1] * absDot01[1][2] + E1[2] * absDot01[1][1];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 1;
			result.separating[1] = 0;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[1]xA1[1].
		r = std::abs(dotDA0[0] * dot01[2][1] - dotDA0[2] * dot01[0][1]);
		r0 = E0[0] * absDot01[2][1] + E0[2] * absDot01[0][1];
		r1 = E1[0] * absDot01[1][2] + E1[2] * absDot01[1][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 1;
			result.separating[1] = 1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[1]xA1[2].
		r = std::abs(dotDA0[0] * dot01[2][2] - dotDA0[2] * dot01[0][2]);
		r0 = E0[0] * absDot01[2][2] + E0[2] * absDot01[0][2];
		r1 = E1[0] * absDot01[1][1] + E1[1] * absDot01[1][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 1;
			result.separating[1] = 2;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[2]xA1[0].
		r = std::abs(dotDA0[1] * dot01[0][0] - dotDA0[0] * dot01[1][0]);
		r0 = E0[0] * absDot01[1][0] + E0[1] * absDot01[0][0];
		r1 = E1[1] * absDot01[2][2] + E1[2] * absDot01[2][1];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 2;
			result.separating[1] = 0;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[2]xA1[1].
		r = std::abs(dotDA0[1] * dot01[0][1] - dotDA0[0] * dot01[1][1]);
		r0 = E0[0] * absDot01[1][1] + E0[1] * absDot01[0][1];
		r1 = E1[0] * absDot01[2][2] + E1[2] * absDot01[2][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 2;
			result.separating[1] = 1;
			return result.intersect;
		}

		// Test for separation on the axis C0 + t*A0[2]xA1[2].
		r = std::abs(dotDA0[1] * dot01[0][2] - dotDA0[0] * dot01[1][2]);
		r0 = E0[0] * absDot01[1][2] + E0[1] * absDot01[0][2];
		r1 = E1[0] * absDot01[2][1] + E1[1] * absDot01[2][0];
		r01 = r0 + r1;
		if (r > r01)
		{
			result.intersect = false;
			result.separating[0] = 2;
			result.separating[1] = 2;
			return result.intersect;
		}

		result.intersect = true;
		return result.intersect;
	}




}	//namespace intersection

}	//namespace ml

#endif
