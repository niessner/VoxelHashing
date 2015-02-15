/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or non-disclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcfacemodule.h"

class PXCFaceData : public PXCBase 
{
public:	
	PXC_CUID_OVERWRITE(PXC_UID('F','D','A','T'));

	enum LandmarkType
	{
		LANDMARK_NOT_NAMED = 0,

		LANDMARK_EYE_RIGHT_CENTER,
		LANDMARK_EYE_LEFT_CENTER,

		LANDMARK_EYELID_RIGHT_TOP,
		LANDMARK_EYELID_RIGHT_BOTTOM,
		LANDMARK_EYELID_RIGHT_RIGHT,
		LANDMARK_EYELID_RIGHT_LEFT,

		LANDMARK_EYELID_LEFT_TOP,
		LANDMARK_EYELID_LEFT_BOTTOM,
		LANDMARK_EYELID_LEFT_RIGHT,
		LANDMARK_EYELID_LEFT_LEFT,

		LANDMARK_EYEBROW_RIGHT_CENTER,
		LANDMARK_EYEBROW_RIGHT_RIGHT,
		LANDMARK_EYEBROW_RIGHT_LEFT,

		LANDMARK_EYEBROW_LEFT_CENTER,
		LANDMARK_EYEBROW_LEFT_RIGHT,
		LANDMARK_EYEBROW_LEFT_LEFT,

		LANDMARK_NOSE_TIP,
		LANDMARK_NOSE_TOP,
		LANDMARK_NOSE_BOTTOM,
		LANDMARK_NOSE_RIGHT,
		LANDMARK_NOSE_LEFT,

		LANDMARK_LIP_RIGHT,
		LANDMARK_LIP_LEFT,

		LANDMARK_UPPER_LIP_CENTER,
		LANDMARK_UPPER_LIP_RIGHT,
		LANDMARK_UPPER_LIP_LEFT,

		LANDMARK_LOWER_LIP_CENTER,
		LANDMARK_LOWER_LIP_RIGHT,
		LANDMARK_LOWER_LIP_LEFT,

		LANDMARK_FACE_BORDER_TOP_RIGHT,
		LANDMARK_FACE_BORDER_TOP_LEFT,

		LANDMARK_CHIN
	};

	enum LandmarksGroupType
	{
		LANDMARK_GROUP_LEFT_EYE = 0x0001,
		LANDMARK_GROUP_RIGHT_EYE = 0x0002,
		LANDMARK_GROUP_RIGHT_EYEBROW = 0x0004,
		LANDMARK_GROUP_LEFT_EYEBROW = 0x0008, 
		LANDMARK_GROUP_NOSE = 0x00010,
		LANDMARK_GROUP_MOUTH = 0x0020,
		LANDMARK_GROUP_JAW = 0x0040
	};

	struct LandmarkPointSource
	{
		pxcI32 index;
		LandmarkType alias;
		pxcI32 reserved[10];
	};

	struct LandmarkPoint
	{
		LandmarkPointSource source;
		pxcI32 confidenceImage;
		pxcI32 confidenceWorld;
	    PXCPoint3DF32 world;
	    PXCPointF32   image;
		pxcI32 reserved[10];
	};
	
	struct HeadPosition
	{
		PXCPoint3DF32 headCenter;
		pxcI32 reserved[10];
	};

	struct PoseEulerAngles
	{		
		pxcF32 yaw;
		pxcF32 pitch;
		pxcF32 roll;
		pxcI32 reserved[10];
	};

	struct PoseQuaternion
	{
		pxcF32 x,y,z,w;
		pxcI32 reserved[10];
	};

	class DetectionData
	{
	public:
		/* 
		* Assigns average depth of detected face to outFaceAverageDepth, returns true if data and outFaceAverageDepth exists, false otherwise. 
		*/
		virtual pxcBool PXCAPI QueryFaceAverageDepth(pxcF32* outFaceAverageDepth) const = 0;
		
		/*
		* Assigns 2D bounding rectangle of detected face to outBoundingRect, returns true if data and outBoundingRect exists, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryBoundingRect(PXCRectI32* outBoundingRect) const = 0;
    protected:		
		virtual ~DetectionData() {}
	};

	class LandmarksData
	{
	public:
		/*
		* Returns the number of tracked landmarks.
		*/
		virtual pxcI32 PXCAPI QueryNumPoints() const = 0;

		/*
		* Assigns all the points to outNumPoints array.
		* App is expected to allocate outPoints array.
		* Returns true if data and parameters exists, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryPoints(LandmarkPoint* outPoints) const = 0;
		
		/*
		* Assigns point matched to index to outPoint.
		* Returns true if data and outPoint exists and index is correct, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryPoint(pxcI32 index, LandmarkPoint* outPoint) const = 0;
		
		/*
		* Returns the number of tracked landmarks in groupFlags.
		*/
		virtual pxcI32 PXCAPI QueryNumPointsByGroup(LandmarksGroupType groupFlags) const = 0;

		/*
		* Assigns points matched to groupFlags to outPoints.
		* User is expected to allocate outPoints to size bigger than the group size - point contains the original source (index + name).
		* Returns true if data and parameters exist, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryPointsByGroup(LandmarksGroupType groupFlags, LandmarkPoint* outPoints) const = 0;
		
		/*
		* Mapping function -> retrieves index corresponding to landmark's name.
		*/
		virtual pxcI32 PXCAPI QueryPointIndex(LandmarkType name) const = 0;
    protected:
		virtual ~LandmarksData() {}
	};

	class PoseData
	{
	public:
		/*
		* Assigns pose Euler angles to outPoseEulerAngles.
		* Returns true if data and parameters exist, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryPoseAngles(PoseEulerAngles* outPoseEulerAngles) const = 0;
		
		/*
		* Assigns pose rotation as quaternion to outPoseQuaternion.
		* Returns true if data and parameters exist, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryPoseQuaternion(PoseQuaternion* outPoseQuaternion) const =0;
		
		/*
		* Assigns Head Position to outHeadPosition.
		* Returns true if data and parameters exist, false otherwise.
		*/
		virtual pxcBool PXCAPI QueryHeadPosition(HeadPosition* outHeadPosition) const = 0;
		
		/*
		* Assigns 3x3 face's rotation matrix to outRotationMatrix.
		* Returns true if data and parameters exist, false otherwise.
		*/		
		virtual pxcBool PXCAPI QueryRotationMatrix(pxcF64 outRotationMatrix[9]) const = 0;

	protected:
		virtual ~PoseData() {}
	};

	class ExpressionsData
	{
	public:
		enum FaceExpression
		{
			EXPRESSION_BROW_RAISER_LEFT = 0,
			EXPRESSION_BROW_RAISER_RIGHT,   
			EXPRESSION_BROW_LOWERER_LEFT,   
			EXPRESSION_BROW_LOWERER_RIGHT,  

			EXPRESSION_SMILE,
			EXPRESSION_KISS,
			EXPRESSION_MOUTH_OPEN,

			EXPRESSION_EYES_CLOSED_LEFT, 
			EXPRESSION_EYES_CLOSED_RIGHT,

			EXPRESSION_HEAD_TURN_LEFT, 
			EXPRESSION_HEAD_TURN_RIGHT,
			EXPRESSION_HEAD_UP, 
			EXPRESSION_HEAD_DOWN,
			EXPRESSION_HEAD_TILT_LEFT,
			EXPRESSION_HEAD_TILT_RIGHT,

			EXPRESSION_EYES_TURN_LEFT, 
			EXPRESSION_EYES_TURN_RIGHT, 
			EXPRESSION_EYES_UP,
			EXPRESSION_EYES_DOWN
		};

		struct FaceExpressionResult
		{
			pxcI32 intensity; // percentage 0 - 100
			pxcI32 reserved[10];
		};

		/** 
			@brief Queries single expression result.
			@param[in] expression requested expression
			@param[out] expressionResult output of expression - for example intensity of expression in frame.
			@return true if expression was calculated successfully, false if expression calculation failed.
		*/		
		virtual pxcBool PXCAPI QueryExpression(FaceExpression expression, FaceExpressionResult* expressionResult) const = 0;

	protected:
		virtual ~ExpressionsData() {}
	};

	class RecognitionData
	{
	public:
		/** 
			@brief Register a user in the Recognition database.
			@param[in] userID The ID of the detected face to be registered.
			@return The unique user ID assigned to the registered face by the Recognition module.
		*/
		virtual pxcUID PXCAPI RegisterUser() = 0;

		/** 
		@brief Removes a user from the Recognition database.
		@param[in] userID The ID of the user to be removed.
		*/
		virtual void PXCAPI UnregisterUser() = 0;

		/** 
			@brief Checks if a user is registered in the Recognition database.
			@param[in] userID The ID of the detected face to search for in the database.
			@return true - if user is in the database, false otherwise.
		*/
		virtual pxcBool PXCAPI IsRegistered() const = 0;

		/** 
			@brief Returns the ID assigned to the current face by the Recognition module
			@return The ID assigned by the Recognition module, or -1 if face was not recognized.
		*/
		virtual pxcUID PXCAPI QueryUserID() const = 0;

	protected:
		virtual ~RecognitionData() {}
	};

	class RecognitionModuleData
	{
	public:
				/** 
			@brief Retrieves the size of the Recognition database for the user to be able to allocate the db buffer in the correct size
			@return The size of the database in bytes.
		*/
		virtual pxcI32 PXCAPI QueryDatabaseSize() const = 0;

		/** 
			@brief Retrieves the Recognition database to allow the user to save the database for later usage.
			@param[in] buffer A user allocated buffer to copy the database into. The user must make sure the buffer is large enough (database size can be determined by calling QueryDatabaseSize()), and deallocated it if necessary.
			@return true if database has been successfully copied to db. false - otherwise.
		*/
		virtual pxcBool PXCAPI QueryDatabaseBuffer(pxcBYTE* buffer) const = 0;

	protected:
		virtual ~RecognitionModuleData() {}
	};

	class Face
	{
	public:
		/* 
		* Returns user ID.
		*/
		virtual pxcUID PXCAPI QueryUserID() const = 0;
		
		/*
		* Returns user's detection data instance - null if it is not enabled.
		*/
		virtual PXCFaceData::DetectionData* PXCAPI QueryDetection() = 0;
		
		/*
		* Returns user's landmarks data instance - null if it is not enabled.
		*/
		virtual PXCFaceData::LandmarksData* PXCAPI QueryLandmarks() = 0;

		/*
		* Returns user's pose data - null if it is not enabled.
		*/
		virtual PXCFaceData::PoseData* PXCAPI QueryPose() = 0;

		/*
		* Returns user's expressions data - null if it not enabled.
		*/
		virtual PXCFaceData::ExpressionsData* PXCAPI QueryExpressions() = 0;
		
		/*
		* Returns user's recognition data - null if it is not enabled.
		*/
		virtual PXCFaceData::RecognitionData* PXCAPI QueryRecognition() = 0;		

     protected:
		virtual ~Face() {}
	};

	/*
	* Updates data to latest available output.
	*/
	virtual pxcStatus PXCAPI Update() = 0;
		
	/*
	* Returns detected frame timestamp.
	*/
	virtual pxcI64 PXCAPI QueryFrameTimestamp() const = 0;

	/*
	* Returns number of total detected faces in frame.
	*/
	virtual pxcI32 PXCAPI QueryNumberOfDetectedFaces() const = 0;
	
	/*
	* Returns tracked face corresponding with the algorithm-assigned faceId, null if no such faceId exists.
	*/
	virtual PXCFaceData::Face* PXCAPI QueryFaceByID(pxcUID faceId) const = 0;

	/*
	* Returns tracked face corresponding with the given index, 0 being the first according the to chosen tracking strategy, null if no such index exists.
	*/
	virtual PXCFaceData::Face* PXCAPI QueryFaceByIndex(pxcI32 index) const = 0;
	
	/*
	* Returns allocated array of numDetectedFaces tracked faces where the order is indicated by heuristic (GetNumberOfDetectedFaces == numDetectedFaces).
	* Size of returned array is guaranteed to be greater or equal to numDetectedFaces.
	*/
	virtual PXCFaceData::Face** PXCAPI QueryFaces(pxcI32* numDetectedFaces) const = 0;

	/*
	* Returns interface for Recognition module data for the entire frame, as opposed to a specific face
	*/
	virtual PXCFaceData::RecognitionModuleData* QueryRecognitionModule() const = 0;

#pragma region
	struct AlertData
	{
		/** @enum AlertType
 		Available events that can be detected by the system (alert types) 
 		*/
 		enum AlertType
		{    
 			ALERT_NEW_FACE_DETECTED = 1,	//  a new face enters the FOV and its position and bounding rectangle is available. 
			ALERT_FACE_OUT_OF_FOV,			//  a new face is out of field of view (even slightly). 
			ALERT_FACE_BACK_TO_FOV,			//  a tracked face is back fully to field of view. 
			ALERT_FACE_OCCLUDED,			//  face is occluded by any object or hand (even slightly).
			ALERT_FACE_NO_LONGER_OCCLUDED,  //  face is not occluded by any object or hand.
			ALERT_FACE_LOST					//  a face could not be detected for too long, will be ignored.
		};
 
 		AlertType	label;	    	// The label that identifies this alert
 		pxcI64      timeStamp;		// The time-stamp in which the event occurred
 		pxcUID      faceId;	    	// The ID of the relevant face, if relevant and known
		pxcI32		reserved[10];
	};

	PXC_DEFINE_CONST(ALERT_NAME_SIZE,30);

	/** 
		@brief Get the details of the fired alert at the requested index.
		@param[in] index the zero-based index of the requested fired alert .
		@param[out] alertData contains all the information for the fired event. 
		@see AlertData
		@note the index is between 0 and the result of GetFiredAlertsNumber()
		@see GetFiredAlertsNumber()
		@return PXC_STATUS_NO_ERROR if returning fired alert data was successful; otherwise, return one of the following errors:
		PXC_STATUS_PROCESS_FAILED - Module failure during processing.
		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
	*/
	virtual pxcStatus PXCAPI QueryFiredAlertData(pxcI32 index, PXCFaceData::AlertData* alertData) const = 0;

	/**
		@brief Return whether the specified alert is fired for a specific face in the current frame, and retrieve its data.
		@param[in] alertEvent the label of the alert event.
		@param[out] outAlertName parameter to contain the name of the alert,maximum size - ALERT_NAME_SIZE
		@see AlertData
		@return PXC_STATUS_NO_ERROR if returning the alert's name was successful; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - if outAlertName is null.
		PXC_STATUS_DATA_UNAVAILABLE - if no alert corresponding to alertEvent was found.
	*/
	virtual pxcStatus PXCAPI QueryAlertNameByID(AlertData::AlertType alertEvent, pxcCHAR* outAlertName) const = 0;

	/**
		@brief Get the number of fired alerts in the current frame.
		@return the number of fired alerts.
	*/
	virtual pxcI32 PXCAPI QueryFiredAlertsNumber(void) const = 0;
        
	/**
		@brief Return whether the specified alert is fired in the current frame, and retrieve its data if it is.
		@param[in] alertEvent the ID of the event.
		@param[out] alertData contains all the information for the fired event.
		@see AlertData
		@return true if the alert is fired, false otherwise.
	*/
	virtual pxcBool PXCAPI IsAlertFired(AlertData::AlertType alertEvent, PXCFaceData::AlertData* alertData) const = 0;

	/**
		@brief Return whether the specified alert is fired for a specific face in the current frame, and retrieve its data.
		@param[in] alertEvent the label of the alert event.
		@param[in] faceID the ID of the face who's alert should be retrieved. 
		@param[out] alertData contains all the information for the fired event.
		@see AlertData
		@return true if the alert is fired, false otherwise.
    */
    virtual pxcBool PXCAPI IsAlertFiredByFace(AlertData::AlertType alertEvent, pxcUID faceID, AlertData* alertData) const = 0;        
	


#pragma  endregion Alerts

};

/* Convenient Operators */
static inline PXCFaceData::LandmarksGroupType operator|(PXCFaceData::LandmarksGroupType a, PXCFaceData::LandmarksGroupType b)
{
	return static_cast<PXCFaceData::LandmarksGroupType>(static_cast<pxcI32>(a) | static_cast<pxcI32>(b));
}

static inline PXCFaceData::AlertData::AlertType operator|(PXCFaceData::AlertData::AlertType a, PXCFaceData::AlertData::AlertType b)
{
	return static_cast<PXCFaceData::AlertData::AlertType>(static_cast<pxcI32>(a) | static_cast<pxcI32>(b));
}