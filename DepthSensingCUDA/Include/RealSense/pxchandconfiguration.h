/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2013-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include "pxchanddata.h"

/**
	@class PXCHandConfiguration
	@brief Handles the setup and configuration of the hand module.
	This interface should be used to configure the tracking, alerts, gestures and output options.
	@note The details of this configuration will be applied only when ApplyChanges is called
*/
class PXCHandConfiguration: public PXCBase
{
public:

	/* Constants */
	PXC_CUID_OVERWRITE(PXC_UID('H','A','C','G'));

	/* Event Handlers */
	
	/**	
		@class AlertHandler
		Interface for callback of alert events 
	*/
	class AlertHandler {
	public:
		/**
		 @brief The OnFiredAlert method is called when a registered alert event is fired.
		 @param[in] alertData contains all the information for the fired event.
		 @see PXCHandData::AlertData
		*/
		virtual void PXCAPI OnFiredAlert(const PXCHandData::AlertData & alertData) = 0;
	};

	/** 
		@class GestureHandler
		Interface for callback of gesture events 
	*/
	class GestureHandler {
	public:

		/**
			 @brief The OnFiredGesture method is called when a registered gesture event is fired.
			 @param[in] gestureData contains all the info of the fired event.
			 @see PXCHandData::GestureData
		*/
		virtual  void PXCAPI OnFiredGesture(const PXCHandData::GestureData & gestureData) = 0;
	};

public:

	/* General */

	/**
		@brief Commit the configuration changes to the module
		This method must be called in order for any configuration changes to actually apply
        @return PXC_STATUS_NO_ERROR - if the operation succeeded 
        PXC_STATUS_DATA_NOT_INITIALIZED - if the configuration was not initialized.\n                        
	*/
	virtual pxcStatus PXCAPI ApplyChanges() = 0;

	/**  
		@brief Restore configuration settings to the default values 
        @return PXC_STATUS_NO_ERROR - if the operation succeeded 
        PXC_STATUS_DATA_NOT_INITIALIZED - if the configuration was not initialized.\n                        
	*/
	virtual pxcStatus PXCAPI RestoreDefaults() = 0;

	/**
		@brief Updates configuration settings to the current state of the module
        @return PXC_STATUS_NO_ERROR - if the operation succeeded 
        PXC_STATUS_DATA_NOT_INITIALIZED - if the configuration was not initialized.\n                        
	*/
	virtual pxcStatus PXCAPI Update() = 0;

	/* Tracking Configuration */
	
    /** 
        @brief Reset all tracking and alert information. 
		For example, you might want to call this method when transitioning from one game level
        to another, in order not to carry information that is not relevant to the new stage.
        @return PXC_STATUS_NO_ERROR if the reset was successful; otherwise, return one of the following errors:
        PXC_STATUS_PROCESS_FAILED - Module failure during processing.\n
        PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.\n                        
        @note ResetTracking will be executed only when processing the next frame
    */
    virtual pxcStatus PXCAPI ResetTracking() = 0;

	/**
		@brief Specify the name of the current user for personalization
		The user name will be used to save and retrieve specific measurements (calibration) for this user
		@param[in] userName the name of the current user
		@return PXC_STATUS_NO_ERROR if the name was set successfully
		PXC_STATUS_PARAM_UNSUPPORTED - if the input user name is an empty string.				
	*/
	virtual pxcStatus PXCAPI SetUserName(const pxcCHAR *userName) = 0;

	/**
		@brief Get the name of the current personalized user.
		@return a null terminating string of the user's name		
	*/
	virtual const pxcCHAR*  PXCAPI QueryUserName() = 0;
	
	/**
		@brief Enable the calculation of speed information for a specific joint
		@param[in] jointLabel the identifier of the joint
		@see PXCHandData::JointType
		@param[in] jointSpeed the identifiers of joint speed type that it moves towards the target
		@see PXCHandData::JointSpeedType
		@param[in] time period for which the average speed will be calculated. Use 0 for current value only (not averaged).
		@return PXC_STATUS_NO_ERROR if joint-speed was enabled successfully; 
		PXC_STATUS_PARAM_UNSUPPORTED if one of the arguments is illegal
	*/
	virtual pxcStatus PXCAPI EnableJointSpeed(PXCHandData::JointType jointLabel, PXCHandData::JointSpeedType jointSpeed, pxcI32 time) = 0;
	
	/**
		@brief Disable the calculation of the speed information for a specific joint
		@param[in] jointType the identifier of the joint
		@see JointType
		@return PXC_STATUS_NO_ERROR if disable joint speed successfully
		PXC_STATUS_PARAM_UNSUPPORTED if the joint was not found
	*/
	virtual pxcStatus PXCAPI DisableJointSpeed(PXCHandData::JointType jointLabel) = 0;		

	/**
		@brief Set the boundaries of the tracking area.
		The boundaries create a frustum shape in which the hand is tracked.
		When the tracked hand reaches one of the boundaries (near, far, left, right, top, or bottom), the appropriate alert is fired.
		@param nearTrackingDistance Tracking bounds frustum: nearest tracking distance
		@param farTrackingDistance Tracking bounds frustum: farthest tracking distance
		@param nearTrackingWidth Tracking bounds frustum: width of tracking range at the nearest distance
		@param nearTrackingHeight Tracking bounds frustum: height of tracking range at the nearest distance
		@return PXC_STATUS_NO_ERROR if the operation succeeded 
		PXC_STATUS_PARAM_UNSUPPORTED if one of the arguments is illegal
	*/
	virtual pxcStatus PXCAPI SetTrackingBounds(pxcF32 nearTrackingDistance, pxcF32 farTrackingDistance, pxcF32 nearTrackingWidth, pxcF32 nearTrackingHeight) = 0;

	/**
		@brief Get the frustum defining the tracking boundaries
		@param[out] nearTrackingDistance Tracking bounds frustum: nearest tracking distance
		@param[out] farTrackingDistance Tracking bounds frustum: farthest tracking distance
		@param[out] nearTrackingWidth Tracking bounds frustum: width of tracking range at the nearest distance
		@param[out] nearTrackingHeight Tracking bounds frustum: height of tracking range at the nearest distance
		@return PXC_STATUS_NO_ERROR if the operation succeeded 
    */
	virtual pxcStatus PXCAPI QueryTrackingBounds(pxcF32& nearTrackingDistance, pxcF32& farTrackingDistance, pxcF32& nearTrackingWidth, pxcF32& nearTrackingHeight) = 0;

	/**
		@brief Set tracking mode which indicates which algorithm will be applied for tracking the hands
		@param trackingMode: the tracking mode to be set
		@Returns PXC_STATUS_NO_ERROR if the tracking mode is set successfully
	*/
	virtual pxcStatus PXCAPI SetTrackingMode(PXCHandData::TrackingModeType trackingMode) = 0;

	/**
		@brief Query tracking mode which indicates which algorithm will be applied for tracking the hands
		@Return TrackingModeType
	*/
	virtual PXCHandData::TrackingModeType  PXCAPI QueryTrackingMode() = 0;

	///@brief <Deprecated> Sets the distance unit of the tracking data
	virtual pxcStatus PXCAPI SetDistanceUnit(PXCHandData::DistanceUnitType distanceUnit) = 0;

	/**
		@brief <Deprecated> Query distance unit of the tracking data
		@ Returns distance unit type
	*/
	virtual PXCHandData::DistanceUnitType  PXCAPI QueryDistanceUnit() = 0;

	/**
		@brief Sets the strength of the smoothing, ranging from 0 (not smoothed) to 1 (very smoothed motion)
		@ Returns PXC_STATUS_NO_ERROR, if the smoothing value is set successfully
		PXC_STATUS_PARAM_UNSUPPORTED if smoothing value is out of range
	*/
	virtual pxcStatus PXCAPI SetSmoothingValue(pxcF32 smoothingValue) = 0;

	/**
		@brief Query smoothing value ranging from 0 (not smoothed) to 1 (very smoothed motion)
		@Return the current smoothing value
	*/
	virtual pxcF32  PXCAPI QuerySmoothingValue() = 0;
	
	/**
		@brief Enable the calculation of normalized skeleton
		Calculating the normalized skeleton applies the pose of the tracked hand to a fixed-size skeleton.
		The positions of the normalized skeleton's joints are available by calling IHand::QueryNormalizedJoint
		@see PXCHandData::IHand::QueryNormalizedJoint
		@param[in] enableFlag flag indicating if the normalized skeleton should be calculated
		@Return PXC_STATUS_NO_ERROR, if the enable flag is set successfully
	*/
	virtual pxcStatus PXCAPI EnableNormalizedJoints(pxcBool enableFlag) = 0;

	/**
	 @brief Return true if normalized joints calculation is enabled, false otherwise
	 @Return true if normalized joints calculation is enabled, false otherwise
	*/
	virtual pxcBool  PXCAPI IsNormalizedJointsEnabled() = 0;

	/**
	 @brief Enable the calculation of the hand segmentation image
	 @param[in] enableFlag flag indicating if the segmentation image should be calculated
	 @Return PXC_STATUS_NO_ERROR, if the enable flag is set successfully
	*/
	virtual pxcStatus PXCAPI EnableSegmentationImage(pxcBool enableFlag) = 0;

	/**
	 @brief Return true if calculation of the hand segmentation image is enabled, false otherwise
	 @Return true if calculation of the hand segmentation image is enabled, false otherwise
	*/
	virtual pxcBool  PXCAPI IsSegmentationImageEnabled() = 0;

	/**
	 @brief Enable the calculation of tracked joints
	 @param[in] enableFlag flag indicating if the joints' tracking should be calculated
	 @ Returns PXC_STATUS_NO_ERROR, if the enable flag is set successfully
	*/
	virtual pxcStatus PXCAPI EnableTrackedJoints(pxcBool enableFlag) = 0;

	/**
	 @brief Return true if calculation of the tracked joints is enabled, false otherwise
	 @Return true if calculation of the tracked joints is enabled, false otherwise
	*/
	virtual pxcBool  PXCAPI IsTrackedJointsEnabled() = 0;

	/* Alerts Configuration */
		
	/** 
		@brief Enable alert messaging for a specific event.            
		@param[in] alertEvent the ID of the event to be enabled.
		@return PXC_STATUS_NO_ERROR if enabling the alert was successful; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter.
	*/
	virtual pxcStatus PXCAPI EnableAlert(PXCHandData::AlertType alertEvent) = 0;
	
	/** 
		@brief Enable all alert messaging events.            
		@return PXC_STATUS_NO_ERROR if enabling all alerts was successful; otherwise, return one of the following errors:
		PXC_STATUS_FEATURE_UNSUPPORTED - Module does not support enabling all alerts
	*/
	virtual pxcStatus PXCAPI EnableAllAlerts(void) = 0;
	
	/** 
		@brief Check if an alert is enabled.    
		@param[in] alertEvent the ID of the event.            
		@return true if the alert is enabled; otherwise, return false
	*/
	virtual pxcBool PXCAPI IsAlertEnabled(PXCHandData::AlertType alertEvent) const = 0;
	
	/** 
		@brief Disable alert messaging for a specific event.            
		@param[in] alertEvent the ID of the event to be disabled.
		@return PXC_STATUS_NO_ERROR if disabling the alert was successful; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter.\n
		PXC_STATUS_DATA_NOT_INITIALIZED - Data was not initialized.\n
	*/
	virtual pxcStatus PXCAPI DisableAlert(PXCHandData::AlertType alertEvent) = 0;

	/** 
		@brief Disable all alerts messaging for all events.                        
		@return PXC_STATUS_NO_ERROR if disabling all alerts was successful; otherwise, return one of the following errors:
		PXC_STATUS_DATA_NOT_INITIALIZED - Data was not initialized.\n
	*/
	virtual pxcStatus PXCAPI DisableAllAlerts(void) = 0;
	
	/** 
		@brief Register an event handler object for the alerts. 
		The event handler's OnFiredAlert method will be called each time an alert is identified.
		@param[in] alertHandler a pointer to the event handler.
		@see AlertHandler::OnFiredAlert
		@return PXC_STATUS_NO_ERROR if the registering an event handler was successful; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input handler is null        
	*/
	virtual pxcStatus PXCAPI SubscribeAlert(AlertHandler *alertHandler) = 0;

	/** 
		@brief Unsubscribe an event handler object for the alerts.
		@param[in] alertHandler a pointer to the event handler that should be removed.
		@return PXC_STATUS_NO_ERROR if the unregistering the event handler was successful, otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input handler is null        
	*/
	virtual pxcStatus PXCAPI UnsubscribeAlert(AlertHandler *alertHandler) = 0;
	
	/* Gestures Configuration */
	
	/** 
		@brief Load a set of gestures from the specified path.
		After this call the gestures that are contained in the gesture-pack will be available for recognition.
		@param[in] gesturePackPath the directory of the gestures.
		@return PXC_STATUS_NO_ERROR if the set of gestures was loaded successfully; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - if the path is empty or gesture name list is empty
	*/
	virtual pxcStatus PXCAPI LoadGesturePack(const pxcCHAR* gesturePackPath)=0;

	 /** 
		@brief Unload sets of gestures from the specified path.          
		@param[in] gesturePackPath the directory of the gestures.
		@return PXC_STATUS_NO_ERROR if all gesture pack were unloaded successfully;  
	*/
	virtual pxcStatus PXCAPI UnloadGesturePack(const pxcCHAR* gesturePackPath) =0;

	 /** 
		@brief Unload all the sets of the gestures.          
		@return PXC_STATUS_NO_ERROR if all gesture packs were unloaded successfully;  
	*/
	virtual pxcStatus PXCAPI UnloadAllGesturesPacks(void)=0;
	
	/**
		@brief Return the total number of gestures that can be recognized.
		@return the total number of gestures that can be recognized.
	*/
	virtual pxcI32 PXCAPI QueryGesturesTotalNumber(void) const = 0;

	/** 
		@brief Retrieve the gesture name that matches the given index.			
		@param[in] index the index of the gesture whose name you want to retrieve.
		@param[in] bufferSize the size of the gestureName buffer.						
		@param[out] gestureName a character buffer to be filled with the gesture name.		
		@return PXC_STATUS_NO_ERROR if the gesture name was retrieved successfully; 
		PXC_STATUS_ITEM_UNAVAILABLE - if there is no corresponding gesture for the given index value
	*/
	virtual pxcStatus PXCAPI QueryGestureNameByIndex(pxcI32 index, pxcI32 bufferSize, pxcCHAR *gestureName) const = 0;
	
	/** 
		@brief Enable a gesture, so that events are fired when the gesture is identified.			
		@param[in] gestureName the name of the gesture to enabled. 
		@param[in] continuousGesture set to "true" to get an event at every frame, or "false" to get only start and end states of the gesture
		@return PXC_STATUS_NO_ERROR if the gesture was enabled successfully; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter.
	*/
	 virtual pxcStatus PXCAPI EnableGesture(const pxcCHAR* gestureName, pxcBool continuousGesture)=0;
	 __inline pxcStatus EnableGesture(const pxcCHAR* gestureName) { return EnableGesture(gestureName, false); }

	/** 
		@brief Enable all gestures, so that events are fired when any gestures is identified.		
		@param[in] continuousGesture set to "true" to get an event at every frame, or "false" to get only start and end states of the gesture
		@return PXC_STATUS_NO_ERROR if all the gestures were enabled successfully;  
	*/
	  virtual pxcStatus PXCAPI EnableAllGestures(pxcBool continuousGesture)=0;
	 __inline pxcStatus EnableAllGestures(void) { return EnableAllGestures(false); }

	/** 
		@brief Check whether a gesture is enabled.
		@param[in] gestureName the name of the gesture.
		@return true if a gesture is enabled, false otherwise.
	*/
	virtual pxcBool PXCAPI IsGestureEnabled(const pxcCHAR* gestureName) const = 0;

	/** 
		@brief Deactivate identification of a gesture. Events will no longer be fired for this gesture.            
		@param[in] gestureName the name of the gesture to deactivate.            
		@return PXC_STATUS_NO_ERROR if the gesture was deactivated successfully; otherwise, return one of the following errors:
		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter value of gestureName
	*/
	virtual pxcStatus PXCAPI DisableGesture(const pxcCHAR* gestureName)=0;

	/** 
		@brief Deactivate identification of all gestures. Events will no longer be fired for any gesture.            
		@return PXC_STATUS_NO_ERROR if the gestures were deactivated successfully; 
	*/
	virtual pxcStatus PXCAPI DisableAllGestures(void)=0;
		   
	/** 
		@brief Register an event handler object for the gestures. 
		The event handler's OnFiredGesture method will be called each time a gesture is identified.
		@param[in] gestureHandler a pointer to the gesture handler.
		@see GestureHandler::OnFiredGesture
		@return PXC_STATUS_NO_ERROR if the subscribe gesture retrieved successfully; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input gesture handler is null.                        
	*/
	virtual pxcStatus PXCAPI SubscribeGesture(GestureHandler* gestureHandler) = 0;

	/** 
		@brief Unsubscribe an event handler object for the gestures.
		@param[in] gestureHandler a pointer to the event handler that should be removed.
		@return PXC_STATUS_NO_ERROR if the subscribe alert successfully,otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED - if the input gesture handler is null.             
	*/
	virtual pxcStatus PXCAPI UnsubscribeGesture(GestureHandler *gestureHandler) = 0;
};
 