/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxcimage.h"

class PXCHandData: public PXCBase
{
public:
	/* Constants */
	PXC_CUID_OVERWRITE(PXC_UID('H','A','D','T')); 
	PXC_DEFINE_CONST(NUMBER_OF_FINGERS,5); 
	PXC_DEFINE_CONST(NUMBER_OF_EXTREMITIES,6);
	PXC_DEFINE_CONST(NUMBER_OF_JOINTS,22);
	PXC_DEFINE_CONST(RESERVED_NUMBER_OF_JOINTS,32);
	PXC_DEFINE_CONST(MAX_NAME_SIZE,64);
	PXC_DEFINE_CONST(MAX_PATH_NAME,256);
	
	/* Enumerations */
	
	/** @enum JointType
		Identifiers of joints that can be tracked by the hand module
	*/
	enum JointType
	{
		/// The center of the wrist
		JOINT_WRIST=0			
		, JOINT_CENTER			/// The center of the palm
		, JOINT_THUMB_BASE		/// Thumb finger joint 1 (base)
		, JOINT_THUMB_JT1		/// Thumb finger joint 2
		, JOINT_THUMB_JT2		/// Thumb finger joint 3
		, JOINT_THUMB_TIP		/// Thumb finger joint 4 (fingertip)
		, JOINT_INDEX_BASE		/// Index finger joint 1 (base)
		, JOINT_INDEX_JT1		/// Index finger joint 2
		, JOINT_INDEX_JT2		/// Index finger joint 3
		, JOINT_INDEX_TIP		/// Index finger joint 4 (fingertip)
		, JOINT_MIDDLE_BASE		/// Middle finger joint 1 (base)
		, JOINT_MIDDLE_JT1		/// Middle finger joint 2
		, JOINT_MIDDLE_JT2		/// Middle finger joint 3
		, JOINT_MIDDLE_TIP		/// Middle finger joint 4 (fingertip)
		, JOINT_RING_BASE		/// Ring finger joint 1 (base)
		, JOINT_RING_JT1		/// Ring finger joint 2
		, JOINT_RING_JT2		/// Ring finger joint 3
		, JOINT_RING_TIP		/// Ring finger joint 4 (fingertip)
		, JOINT_PINKY_BASE		/// Pinky finger joint 1 (base)
		, JOINT_PINKY_JT1		/// Pinky finger joint 2
		, JOINT_PINKY_JT2		/// Pinky finger joint 3
		, JOINT_PINKY_TIP		/// Pinky finger joint 4 (fingertip)		
	};

	/**
		@enum ExtremityType
		The identifier of an extremity of the tracked hand
	*/
	enum ExtremityType {
		/// The closest point to the camera in the tracked hand
		EXTREMITY_CLOSEST=0 
		, EXTREMITY_LEFTMOST 		/// The left-most point of the tracked hand
		, EXTREMITY_RIGHTMOST		/// The right-most point of the tracked hand 
		, EXTREMITY_TOPMOST			/// The top-most point of the tracked hand
		, EXTREMITY_BOTTOMMOST		/// The bottom-most point of the tracked hand
		, EXTREMITY_CENTER			/// The center point of the tracked hand			
	};

	/** @enum FingerType
		The identifiers of the hand fingers
	*/
	enum FingerType {            
		/// Thumb finger
		FINGER_THUMB=0        
		, FINGER_INDEX           /// Index finger  
		, FINGER_MIDDLE          /// Middle finger
		, FINGER_RING            /// Ring finger
		, FINGER_PINKY           /// Pinky finger
	};
       
	/** @enum BodySideType
		Defines the side of the body that a hand belongs to
	*/
	enum BodySideType {            
		/// The hand-type was not determined    
		BODY_SIDE_UNKNOWN=0        
		, BODY_SIDE_LEFT            /// Left side of the body    
		, BODY_SIDE_RIGHT           /// Right side of the body
	};

	/** @enum AlertType
		Enumerates the events that can be detected and fired by the module
	*/
	enum AlertType {
		 ///  A hand is identified and its mask is available  
		 ALERT_HAND_DETECTED				= 0x0001         
		, ALERT_HAND_NOT_DETECTED			= 0x0002		///  A previously detected hand is lost, either because it left the field of view or because it is occluded
		, ALERT_HAND_TRACKED				= 0x0004		///  Full tracking information is available for a hand
		, ALERT_HAND_NOT_TRACKED			= 0x0008		///  No tracking information is available for a hand (none of the joints are tracked)
		, ALERT_HAND_CALIBRATED				= 0x0010		///  Hand measurements are ready and accurate 
		, ALERT_HAND_NOT_CALIBRATED			= 0x0020		///  Hand measurements are not yet finalized, and are not fully accurate
		, ALERT_HAND_OUT_OF_BORDERS			= 0x0040		///  Hand is outside of the tracking boundaries
		, ALERT_HAND_INSIDE_BORDERS			= 0x0080		///  Hand has moved back inside the tracking boundaries         
		, ALERT_HAND_OUT_OF_LEFT_BORDER		= 0x0100		///  The tracked object is touching the left border of the field of view
		, ALERT_HAND_OUT_OF_RIGHT_BORDER	= 0x0200		///  The tracked object is touching the right border of the field of view
		, ALERT_HAND_OUT_OF_TOP_BORDER		= 0x0400		///  The tracked object is touching the upper border of the field of view
		, ALERT_HAND_OUT_OF_BOTTOM_BORDER	= 0x0800		///  The tracked object is touching the lower border of the field of view
		, ALERT_HAND_TOO_FAR				= 0x1000		///  The tracked object is too far
		, ALERT_HAND_TOO_CLOSE				= 0x2000		///  The tracked object is too close		
	};
	


	/** 
		@enum GestureStateType
		Available gesture event states
	*/
	enum GestureStateType {
		/// Gesture started
		GESTURE_STATE_START=0			 
		, GESTURE_STATE_IN_PROGRESS	/// Gesture is in progress
		, GESTURE_STATE_END			/// Gesture ended
	};
		       
	/** 
		@enum TrackingModeType
		The Tracking mode indicates which set of joints will be tracked.
	*/
	enum TrackingModeType { 
		/// Track the full skeleton	
		TRACKING_MODE_FULL_HAND=0	
		, TRACKING_MODE_EXTREMITIES	///<Unsupported> Track the extremities of the hand
	};

	/** 
		@enum DistanceUnitType
		<Deprecated> allows to choose between the units of the output world distances
	*/
	enum DistanceUnitType
	{
		/// Display output distances in meters
		DISTANCE_UNIT_METERS=0
		,DISTANCE_UNIT_CENTIMETERS	///Display output distances in centimeters
	};

	/** 
		@enum JointSpeedType
		List of available modes for calculating the joint's speed	
	*/
	enum JointSpeedType {
		/// Average speed across time
		JOINT_SPEED_AVERAGE=0
		, JOINT_SPEED_ABSOLUTE	/// Average of absolute speed across time 	
	};

	/** 
		@enum AccessOrderType
		List of the different orders in which the hands can be accessed
	*/
	enum AccessOrderType {
		/// Unique ID of the hand
		ACCESS_ORDER_BY_ID=0			
		, ACCESS_ORDER_BY_TIME 			/// From oldest to newest hand in the scene           
		, ACCESS_ORDER_NEAR_TO_FAR		/// From near to far hand in scene
		, ACCESS_ORDER_LEFT_HANDS		/// All left hands
		, ACCESS_ORDER_RIGHT_HANDS		/// All right hands
		, ACCESS_ORDER_FIXED			/// The index of each hand is fixed as long as it is detected (and between 0 and 1)
	};	

	/* Data Structures */
    
	/** @struct JointData
		Contains the information about the position and rotation of a joint in the hand's skeleton
	*/
	struct JointData 
	{
		pxcI32			confidence;          /// The confidence score of the tracking data, ranging from 0 to 100
		PXCPoint3DF32   positionWorld;       /// The geometric position in world coordinates (meters)
        PXCPoint3DF32   positionImage;       /// The geometric position in depth coordinates (pixels)
        PXCPoint4DF32   localRotation;       /// A quaternion representing the local 3D orientation (relative to parent joint) from the joint's parent to the joint
        PXCPoint4DF32   globalOrientation;   /// A quaternion representing the global 3D orientation (relative to camera) from the joint's parent to the joint
		PXCPoint3DF32	speed;				 /// The speed of the joints in the 3D world coordinates
    };

	/** 
	    @struct ExtremitiesData
		Contains the parameters that define extremities points
    */
	struct ExtremityData 
	{	
		PXCPoint3DF32	pointWorld;		/// 3D world coordinates of the extremity point
		PXCPoint3DF32	pointImage;		/// 2D image coordinates of the extremity point
	};

	/** 
	    @struct FingerData
		Contains the parameters that define a finger
    */
	struct  FingerData
	{
		pxcI32 foldedness;			/// The degree of foldedness of the tracking finger, ranging from 0 to 100
		pxcF32 radius;				/// The radius of the tracked fingertip
	};

	/** 
		@struct AlertData
		Containing the parameters that define an alert
    */
    struct AlertData 
	{
		AlertType	label;	    	/// The label that identifies this alert
		pxcUID      handId;	    	/// The ID of the relevant hand, if relevant and known
		pxcI64      timeStamp;		/// The time-stamp in which the event occurred
		pxcI32      frameNumber;    /// The number of the frame in which the event occurred
	};
	
	/** 
		@struct GestureData
		Contains the parameters that define a gesture
		Default gestures: 
			"spreadfingers"  - hand open facing the camera
			"thumbup" - hand closed with thumb finger pointing up
			"thumbdown"  - hand closed with thumb finger pointing down
			"twofingerspinch"  - hand open with thumb finger and index finger touching each other
			"v_sign" - hand closed with index finger and middle finger pointing up
	*/
	struct GestureData 
	{
		pxcI64				timeStamp;					/// Time-stamp in which the gesture occurred
		pxcUID				handId;	    				/// ID of the relevant tracked hand, if relevant and known
		GestureStateType	state;						/// The state of the gesture			
		pxcI32				frameNumber;				/// The number of the frame in which the gesture occurred			
		pxcCHAR				name[MAX_NAME_SIZE];		/// Unique name of this gesture 		
	};

	/* Interfaces */

    /** 
		@class IHand
		Contains the parameters that define a hand
    */
    class IHand
	{
	public:

	    /**	@brief the hand's unique identifier
	    	@return the hand's unique identifier
		*/
		virtual pxcUID PXCAPI QueryUniqueId() const = 0; 

	    /** 		
		@brief <Reserved> Return the identifier of the user whose hand is represented
		*/
		virtual pxcUID PXCAPI QueryUserId() const = 0; 

	    /** 		
		@brief Return the time-stamp in which the collection of the hand data was completed
		*/
		virtual pxcI64 PXCAPI QueryTimeStamp() const = 0; 

		/**
			@brief Return true if there is a valid hand calibration.
			A valid calibration results in more accurate tracking data, that is better fitted to the user's hand.
		*/
		virtual pxcBool PXCAPI IsCalibrated(void) const = 0;

	    /** 		
		@brief Return the side of the body that the hand belongs to
		*/
		virtual BodySideType PXCAPI QueryBodySide() const = 0; 
		
		/** 		
		@brief Return the location and dimensions of the tracked hand, represented by a 2D bounding box (pixels)
		*/
		virtual const PXCRectI32& PXCAPI QueryBoundingBoxImage() const = 0; 

		/** 		
		@brief Return the 2D center of mass of the hand in image space (pixels) 
		*/
	   virtual const PXCPointF32& PXCAPI QueryMassCenterImage() const = 0; 

	    /** 		
		@brief Return the 3D center of mass of the hand in world space (meters)
		*/
		virtual const PXCPoint3DF32& PXCAPI QueryMassCenterWorld() const = 0; 

	    /** 		
		@brief A quaternion representing the global 3D orientation of center joint
		*/
		virtual const PXCPoint4DF32& PXCAPI QueryPalmOrientation() const = 0; 
		
	    /** 		
		@brief Return the level of openness of the hand
		@return openness level ranging from 0 (all fingers completely folded) to 100 (all fingers fully spread) 
		*/
		virtual pxcI32 PXCAPI QueryOpenness() const = 0;

		/** 		
		@brief Get detected extremities points data
		@param[in] extremityLabel The id of this ExtremityType
		@param[out] extremityPoint The extremity point data
		@return PXC_STATUS_NO_ERROR for success, an error code otherwise
		*/
		virtual pxcStatus PXCAPI QueryExtremityPoint(ExtremityType extremityLabel, ExtremityData& extremityPoint) const = 0; 

		/** 
		@brief Get fingers data
		@param[in] fingerLabel The id of this finger
		@param[out] fingerData The finger data
		@return PXC_STATUS_NO_ERROR for success, an error code otherwise
		*/
		virtual pxcStatus PXCAPI QueryFingerData(FingerType fingerLabel, FingerData& fingerData) const = 0; 

		/** 
		@brief Get tracked hand joint data
		@param[in] jointLabel The id of this joint
		@param[out] jointData The tracked hand joint data
		@return PXC_STATUS_NO_ERROR for success, an error code otherwise
		*/			
		virtual pxcStatus PXCAPI QueryTrackedJoint(JointType jointLabel, JointData& jointData) const = 0; 

		/** 
		@brief Get normalized hand joint data
		@param[in] jointLabel The id of this joint
		@param[out] jointData The normalized hand joint data
		@return PXC_STATUS_NO_ERROR for success, an error code otherwise
		*/	
		virtual pxcStatus PXCAPI QueryNormalizedJoint(JointType jointLabel, JointData& jointData) const = 0; 

		/**			
			@brief Retrieve the 2D image mask of the tracked hand. 	 
			In the image mask, each pixel occupied by the hand is white (value of 255) and all other pixels are black (value of 0).
			@param[out] image the blob image to be returned
			@return PXC_STATUS_NO_ERROR if a current image exists and could be copied; otherwise, return the following error:
			PXC_STATUS_DATA_UNAVAILABLE - if segmentation image is not available.		
		*/		
		virtual pxcStatus PXCAPI QuerySegmentationImage(PXCImage* & image) const = 0; 
		
		/** 
		@brief  Return true/false if tracked joints of HandData exists 
		*/
		virtual pxcBool PXCAPI HasTrackedJoints() const= 0;

		/** 
		@brief  Return true/false if normalized joints of HandData exists 
		*/
		virtual pxcBool PXCAPI HasNormalizedJoints() const= 0;

		/** 
		@brief  Return true/false if hand segmentation image exists 
		*/
		virtual pxcBool PXCAPI HasSegmentationImage()const= 0;

	};	// class IHand

public:
	/* General */

	/**
	* @brief Updates data to latest available output.
	*/
	virtual pxcStatus PXCAPI Update() = 0;

	/* Alerts Outputs */
	
	/**
		@brief Get the number of fired alerts in the current frame.
		@return the number of fired alerts.
	*/
	virtual pxcI32 PXCAPI QueryFiredAlertsNumber(void) const = 0;

	/** 
		@brief Get the details of the fired alert at the requested index.
		@param[in] index the zero-based index of the requested fired alert .
		@param[out] alertData contains all the information for the fired event. 
		@see AlertData
		@note the index is between 0 and the result of QueryFiredAlertsNumber()
		@see QueryFiredAlertsNumber()
		@return PXC_STATUS_NO_ERROR if returning fired alert data was successful; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input parameter value is not supported. For instance, index >= size of all enabled alerts
	*/
	virtual pxcStatus PXCAPI QueryFiredAlertData(pxcI32 index, AlertData & alertData) const = 0;
	
	/**
		@brief Return whether the specified alert is fired in the current frame, and retrieve its data if it is.
		@param[in] alertEvent the ID of the event.
		@param[out] alertData contains all the information for the fired event.
		@see AlertData
		@return true if the alert is fired, false otherwise.
	*/
	virtual pxcBool PXCAPI IsAlertFired(AlertType alertEvent, AlertData & alertData) const = 0;

	/**
		@brief Return whether the specified alert is fired for a specific hand in the current frame, and retrieve its data.
		@param[in] alertEvent the label of the alert event.
		@param[in] handID the ID of the hand who's alert should be retrieved. 
		@param[out] alertData contains all the information for the fired event.
		@see AlertData
		@return true if the alert is fired, false otherwise.
	*/
	virtual pxcBool PXCAPI IsAlertFiredByHand(AlertType alertEvent, pxcUID handID, AlertData & alertData) const = 0;

    /* Gestures Outputs */

	/** 
		@brief Get the number of fired gestures in the current frame.
		@return number of fired gestures.
	*/
	virtual pxcI32 PXCAPI QueryFiredGesturesNumber(void) const = 0;

	/** 
		@brief Get the details of the fired gesture at the requested index.
		@param[in] index the zero-based index of the requested fired gesture.
		@param[out] gestureData contains all the information for the fired gesture.
		@see GestureData
		@note the index is between 0 and the result of QueryFiredGesturesNumber()
		@see QueryFiredGesturesNumber()
		@return PXC_STATUS_NO_ERROR if the fired gesture data successfully; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input parameter value is not supported. For instance, index >= size of all enabled gestures
	*/
	virtual pxcStatus PXCAPI QueryFiredGestureData(pxcI32 index, GestureData & gestureData) const = 0;

	/** 
		@brief Check whether a gesture was fired and return its details if it was.
		@param[in] gestureName the name of the gesture to be checked.
		@param[out] gestureData will contain all the information for the fired gesture.
		@see GestureData
		@return true if the gesture was fired, false otherwise.
	*/
	virtual pxcBool PXCAPI IsGestureFired(const pxcCHAR* gestureName, GestureData & gestureData) const = 0;
        
	/**
		@brief Return whether the specified gesture is fired for a specific hand in the current frame, and retrieve its data.
		@param[in] gestureName the name of the gesture to be checked.
		@param[in] handID the ID of the hand who's alert should be retrieved. 
		@param[out] gestureData will contain all the information for the fired gesture.
		@see GestureData
		@return true if the gesture was fired, false otherwise.
	*/
	virtual pxcBool PXCAPI IsGestureFiredByHand(const pxcCHAR* gestureName, pxcUID handID, GestureData & gestureData) const = 0;

    /* Hands Outputs */
		
	/** 
		@brief Return the number of hands detected in the current frame.            
		@return The number of hands detected in the current frame.
	*/
	virtual pxcI32 PXCAPI QueryNumberOfHands(void) const = 0;

	/** 
		@brief Retrieve the hand's uniqueId.			
		@param[in] accessOrder the order in which the hands are enumerated (accessed).
		@see AccessOrderType
		@param[in] index the index of the hand to be retrieve, based on the given AccessOrder.
		@param[out] handId contains the hand's uniqueId.
		@return PXC_STATUS_NO_ERROR if the hand exists; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter value. For instance, index >= total number of hands					
	*/
	virtual pxcStatus PXCAPI QueryHandId(AccessOrderType accessOrder, pxcI32 index, pxcUID &handId) const = 0;

	/** 
		@brief Retrieve the hand object data using a specific AccessOrder and index in that order
		@param[in] accessOrder the order in which the hands are enumerated (accessed).
		@see AccessOrder
		@param[in] index the index of the hand to be retrieve, based on the given AccessOrder.
		@param[out] hand contains all the information for the hand.
		@see HandData
		@return PXC_STATUS_NO_ERROR if the hand was retrieved successfully; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - if index >= MAX_NUM_HANDS
		PXC_STATUS_DATA_UNAVAILABLE  - if index >= number of detected hands                 
	*/
	virtual pxcStatus PXCAPI QueryHandData(AccessOrderType accessOrder, pxcI32 index, IHand *& handData) const = 0;

	/** 
		@brief Retrieve the hand object data using its unique Id
		@param[in] handID the unique ID of the requested hand
		@param[out] hand contains all the information for the hand.
		@see HandData
		@return PXC_STATUS_NO_ERROR if the hand was retrieved successfully; otherwise, return one of the following errors:
		PXC_STATUS_DATA_UNAVAILABLE  - if there is no output hand data
		PXC_STATUS_PARAM_UNSUPPORTED - if there is no hand data for the given hand ID.                        
	*/
	 virtual pxcStatus PXCAPI QueryHandDataById(pxcUID handID, IHand *& handData) const = 0;
};

/** Operator | for alertType labels */
static inline PXCHandData::AlertType operator|(PXCHandData::AlertType a, PXCHandData::AlertType b)
{
	return static_cast<PXCHandData::AlertType>(static_cast<pxcI32>(a) | static_cast<pxcI32>(b));
}

