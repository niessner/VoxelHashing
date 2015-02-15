
#ifndef CORE_GRAPHICS_CAMERA_TRACKBALL_H_
#define CORE_GRAPHICS_CAMERA_TRACKBALL_H_

namespace ml {

template <class FloatType>
class CameraTrackball : public Camera<FloatType> {
public:
	CameraTrackball() {}
	CameraTrackball(const point3d<FloatType>& eye, const point3d<FloatType>& worldUp, const point3d<FloatType>& right, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) :
		Camera<FloatType>(eye, worldUp, right, fieldOfView, aspect, zNear, zFar)
	{
		m_modelTranslation = point3d<FloatType>::origin;
		m_modelRotation.setIdentity();

		update();
	}

	//! zoom
	void moveModel(FloatType delta) {
		m_modelTranslation += getLook() * delta;
		update();
	}
	//! rotate
	void rotateModel(const point2d<FloatType>& delta) {
		m_modelRotation = mat4f::rotation(getUp(), delta.x) * mat4f::rotation(getRight(), delta.y) * m_modelRotation;
		update();
	}
	void rotateModelUp(FloatType delta) {
		m_modelRotation = mat4f::rotation(getUp(), delta) * m_modelRotation;
		update();
	}
	void rotateModelRight(FloatType delta) {
		m_modelRotation = mat4f::rotation(getRight(), delta) * m_modelRotation;
		update();
	}

	void setModelTranslation(const point3d<FloatType>& t) {
		m_modelTranslation = t;
		update();
	}

	const point3d<FloatType>& getModelTranslation() const { return m_modelTranslation; }
	const Matrix4x4<FloatType>& getModelRotation() const { return m_modelRotation; }

	const Matrix4x4<FloatType>& getWorldViewProj() const { return m_worldViewProj; }
	const Matrix4x4<FloatType>& getWorldView() const { return m_worldView; }
	const Matrix4x4<FloatType>& getWorld() const { return m_world; }

private:
	void update() {
		m_world = mat4f::translation(m_modelTranslation) * m_modelRotation;
		m_worldView = camera() * m_world;
		m_worldViewProj = perspective() * m_worldView;
	}

	point3d<FloatType> m_modelTranslation;
	Matrix4x4<FloatType> m_modelRotation;

	Matrix4x4<FloatType> m_worldViewProj;
	Matrix4x4<FloatType> m_worldView;
	Matrix4x4<FloatType> m_world;
};

typedef CameraTrackball<float> CameraTrackballf;
typedef CameraTrackball<double> CameraTrackballd;

}  // namespace ml


#endif  // CORE_GRAPHICS_CAMERA_H_
