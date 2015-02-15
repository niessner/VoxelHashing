
#ifndef CORE_GRAPHICS_CAMERA_H_
#define CORE_GRAPHICS_CAMERA_H_

namespace ml {

template <class FloatType>
class Camera {
public:
	Camera() {}
    Camera(const std::string &s);
	Camera(const point3d<FloatType>& eye, const point3d<FloatType>& worldUp, const point3d<FloatType>& right, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar);

	//! Construct camera from extrinsics matrix m (columns are x, y, z vectors and origin of camera in that order).
	//! If flipRight is set, flip x to correct for sensor horizontal flipping
	Camera(const Matrix4x4<FloatType>& m, const FloatType fieldOfView, const FloatType aspect, const FloatType zNear, const FloatType zFar, const bool flipRight = false);

	void updateAspectRatio(FloatType newAspect);
	void lookRight(FloatType theta);
	void lookUp(FloatType theta);
	void roll(FloatType theta);

	void strafe(FloatType delta);
	void jump(FloatType delta);
	void move(FloatType delta);

    Ray<FloatType> getScreenRay(FloatType screenX, FloatType screenY) const;
    point3d<FloatType> getScreenRayDirection(FloatType screenX, FloatType screenY) const;

	Matrix4x4<FloatType> camera() const {
		return m_camera;
	}

	void setCamera(Matrix4x4<FloatType>& c){
		m_camera = c;
	}

	Matrix4x4<FloatType> perspective() const {
		return m_perspective;
	}

	Matrix4x4<FloatType> cameraPerspective() const {
		return m_cameraPerspective;
	}

	point3d<FloatType> getEye() const {
		return m_eye;
	}

  point3d<FloatType> getLook() const {
    return m_look;
  }

  point3d<FloatType> getUp() const {
    return m_up;
  }

	FloatType getFoV() const {
		return m_fieldOfView;
	}

	FloatType getAspect() const {
		return m_aspect;
	}

    std::string toString() const;

	void applyTransform(const Matrix4x4<FloatType>& transform);
private:
	void update();
	Matrix4x4<FloatType> perspectiveFov(FloatType fieldOfView, FloatType aspectRatio, FloatType zNear, FloatType zFar);
	Matrix4x4<FloatType> viewMatrix(const point3d<FloatType>& eye, const point3d<FloatType>& look, const point3d<FloatType>& up, const point3d<FloatType>& right);

	point3d<FloatType> m_eye, m_right, m_look, m_up;
	point3d<FloatType> m_worldUp;
	Matrix4x4<FloatType> m_camera;
	Matrix4x4<FloatType> m_perspective;
	Matrix4x4<FloatType> m_cameraPerspective;

	FloatType m_fieldOfView, m_aspect, m_zNear, m_zFar;
};

typedef Camera<float> Cameraf;
typedef Camera<double> Camerad;

}  // namespace ml

#include "camera.cpp"

#endif  // CORE_GRAPHICS_CAMERA_H_
