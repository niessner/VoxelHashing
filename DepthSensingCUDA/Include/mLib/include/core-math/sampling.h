#ifndef _CORE_SAMPLING_H_
#define _CORE_SAMPLING_H_

namespace ml {

	//from mutsuba and prt
	template<class FloatType>
	class Sample {
	public:
		static inline point3d<FloatType> squareToUniformSphere(const point2d<FloatType>& sample) {
			FloatType z = 1 - 2*sample.y;
			FloatType r = std::sqrt(1 - z*z);
			FloatType sinPhi, cosPhi;
			math::sincos(2*(FloatType)math::PI * sample.x, sinPhi, cosPhi);
			return point3d<FloatType>(r*cosPhi, r*sinPhi, z);
		}

		static inline point3d<FloatType> squareToUniformHemisphere(const point2d<FloatType>& sample) {
			FloatType z = sample.x;
			FloatType tmp = std::sqrt(1 - z*z);
			FloatType sinPhi, cosPhi;
			math::sincos(2*(FloatType)math::PI * sample.y, sinPhi, cosPhi);
			return point3d<FloatType>(cosPhi * tmp, sinPhi * tmp, z);
		}


		static inline point3d<FloatType> squareToCosineHemisphere(const point2d<FloatType> &sample) {
			point2d<FloatType> p = squareToUniformDiskConcentric(sample);
			FloatType z = std::sqrt(1 - p.x*p.x - p.y*p.y);
			///* Guard against numerical imprecisions */
			//if (EXPECT_NOT_TAKEN(z == 0))
			//	z = 1e-10f;
			return point3d<FloatType>(p.x, p.y, z);
		}

		static inline point3d<FloatType> squareToPowerCosineSampleHemisphere(const point2d<FloatType> &sample, FloatType exp) {
			//const FloatType theta = 2*(FloatType)math::PI*sample.x;
			//FloatType phi = std::acos(std::pow(sample.y,1/(exp+1)));
			//return point3d<FloatType>(std::cos(theta)*std::sin(phi), std::sin(theta)*std::sin(phi), std::cos(phi));
			
			const FloatType phi = 2*(FloatType)math::PI * sample.x;
			const FloatType cosTheta = std::pow(sample.y,1/(exp+1));
			FloatType sinTheta = std::sqrt(1 - (cosTheta*cosTheta));
			return point3d<FloatType>(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
		}


		//!cosCutoff in cos(\alpha)
		static inline point3d<FloatType> squareToUniformCone(const point2d<FloatType>& sample, FloatType cosCutoff) {
			FloatType cosTheta = (1-sample.x) + sample.x*cosCutoff;
			FloatType sinTheta = std::sqrt(1 - cosTheta*cosTheta);

			FloatType sinPhi, cosPhi;
			math::sincos(2*(FloatType)math::PI * sample.y, &sinPhi, &cosPhi);

			return point3d<FloatType>(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
		}

		static inline point2d<FloatType> squareToUniformDisk(const point2d<FloatType> &sample) {
			FloatType r = std::sqrt(sample.x);
			FloatType sinPhi, cosPhi;
			math::sincos(2*(FloatType)math::PI * sample.y, &sinPhi, &cosPhi);

			return point2d<FloatType>(cosPhi*r, sinPhi*r);
		}

		static inline point2d<FloatType> squareToUniformTriangle(const point2d<FloatType> &sample) {
			FloatType a = std::sqrt(1 - sample.x);
			return point2d<FloatType>(1 - a, a * sample.y);
		}

		static inline point2d<FloatType> squareToUniformDiskConcentric(const point2d<FloatType> &sample) {
			FloatType r1 = 2.0f*sample.x - 1.0f;
			FloatType r2 = 2.0f*sample.y - 1.0f;

			/* Modified concentric map code with less branching (by Dave Cline), see
			http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
			FloatType phi, r;
			if (r1 == 0 && r2 == 0) {
				r = phi = 0;
			} else if (r1*r1 > r2*r2) {
				r = r1;
				phi = ((FloatType)math::PI/4.0f) * (r2/r1);
			} else {
				r = r2;
				phi = ((FloatType)math::PI/2.0f) - (r1/r2) * ((FloatType)math::PI/4.0f);
			}

			FloatType cosPhi, sinPhi;
			math::sincos(phi, &sinPhi, &cosPhi);

			return point2d<FloatType>(r * cosPhi, r * sinPhi);
		}

		static inline point2d<FloatType> uniformDiskToSquareConcentric(const point2d<FloatType> &p) {
			FloatType r   = std::sqrt(p.x * p.x + p.y * p.y),
				phi = std::atan2(p.y, p.x),
				a, b;

			if (phi < -(FloatType)math::PI/4) {
				/* in range [-pi/4,7pi/4] */
				phi += 2*(FloatType)math::PI;
			}

			if (phi < (FloatType)math::PI/4) { /* region 1 */
				a = r;
				b = phi * a / ((FloatType)math::PI/4);
			} else if (phi < 3*(FloatType)math::PI/4) { /* region 2 */
				b = r;
				a = -(phi - (FloatType)math::PI/2) * b / ((FloatType)math::PI/4);
			} else if (phi < 5*(FloatType)math::PI/4) { /* region 3 */
				a = -r;
				b = (phi - (FloatType)math::PI) * a / ((FloatType)math::PI/4);
			} else { /* region 4 */
				b = -r;
				a = -(phi - 3*(FloatType)math::PI/2) * b / ((FloatType)math::PI/4);
			}

			return point2d<FloatType>(0.5f * (a+1), 0.5f * (b+1));
		}

		static inline point2d<FloatType> squareToStdNormal(const point2d<FloatType> &sample) {
			FloatType r   = std::sqrt(-2 * std::log(1-sample.x)),
				phi = 2 * (FloatType)math::PI * sample.y;
			point2d<FloatType> result;
			math::sincos(phi, &result.y, &result.x);
			return result * r;
		}

		static inline FloatType squareToStdNormalPdf(const point2d<FloatType> &pos) {
			return INV_TWOPI * math::fastexp(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
		}

		static inline FloatType intervalToTent(FloatType sample) {
			FloatType sign;

			if (sample < 0.5f) {
				sign = 1;
				sample *= 2;
			} else {
				sign = -1;
				sample = 2 * (sample - 0.5f);
			}

			return sign * (1 - std::sqrt(sample));
		}

		static inline point2d<FloatType> squareToTent(const point2d<FloatType> &sample) {
			return point2d<FloatType>(
				intervalToTent(sample.x),
				intervalToTent(sample.y)
				);
		}

		static inline FloatType intervalToNonuniformTent(FloatType a, FloatType b, FloatType c, FloatType sample) {
			FloatType factor;

			if (sample * (c-a) < b-a) {
				factor = a-b;
				sample *= (a-c)/(a-b);
			} else {
				factor = c-b;
				sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
			}

			return b + factor * (1-std::sqrt(sample));
		}

	private:
	};

	typedef Sample<float> Samplef;
	typedef Sample<double> Sampled;

} // ml

#endif