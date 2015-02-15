/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or non-disclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#pragma once
#include <pxcbase.h>
#include "pxcfacedata.h"

class PXCFaceConfiguration : public PXCBase
{
public:
	PXC_CUID_OVERWRITE(PXC_UID('F','C','F','G'));

	enum TrackingStrategyType
	{
		STRATEGY_APPEARANCE_TIME,
		STRATEGY_CLOSEST_TO_FARTHEST,
		STRATEGY_FARTHEST_TO_CLOSEST,
		STRATEGY_LEFT_TO_RIGHT,
		STRATEGY_RIGHT_TO_LEFT
	};

	enum SmoothingLevelType
	{
		SMOOTHING_DISABLED,
		SMOOTHING_MEDIUM,
		SMOOTHING_HIGH
	};

	struct DetectionConfiguration
	{
		pxcBool isEnabled;
		pxcI32 maxTrackedFaces;
		SmoothingLevelType smoothingLevel;		
		pxcI32 reserved[10];		
	};

	struct LandmarksConfiguration
	{
		pxcBool isEnabled;
		pxcI32 maxTrackedFaces;
		SmoothingLevelType smoothingLevel;
		pxcI32 numLandmarks;
		pxcI32 reserved[10];
	};

	struct PoseConfiguration
	{
		pxcBool isEnabled;
		pxcI32 maxTrackedFaces;
		SmoothingLevelType smoothingLevel;
		pxcI32 reserved[10];
	};

	class ExpressionsConfiguration
	{
	public:
		struct ExpressionsProperties
		{
			pxcBool isEnabled;
			pxcI32 maxTrackedFaces;
			pxcI32 reserved[10];
		};
		ExpressionsProperties properties;

 		/*
 			@brief Enables expression module.
 		*/
		__inline void Enable()
		{
			properties.isEnabled = true;
		}
 		/*
 			@brief Disables expression module.
 		*/
		__inline void Disable()
		{
			properties.isEnabled = false;
		}
 		/*
 			@brief Is expression module enabled.
			@return true - enabled, false - disabled.
 		*/
		__inline pxcBool IsEnabled()
		{
			return properties.isEnabled;
		}
 		/*
 			@brief Enables all available expressions.
 		*/
		virtual void PXCAPI EnableAllExpressions() = 0;
 		/*
 			@brief Disables all available expressions.
 		*/
		virtual void PXCAPI DisableAllExpressions() = 0;
 		/*
 			@brief Enables specific expression.
			@param[in] expression - single face expression.
			@return PXC_STATUS_NO_ERROR - success.
			PXC_STATUS_PARAM_UNSUPPORTED - expression is unsupported.
 		*/
		virtual pxcStatus PXCAPI EnableExpression(PXCFaceData::ExpressionsData::FaceExpression expression) = 0;
 		/*
 			@brief Disables specific expression.
			@param[in] expression - single face expression.
 		*/
		virtual void PXCAPI DisableExpression(PXCFaceData::ExpressionsData::FaceExpression expression) = 0;
 		/*
 			@brief Checks if expression is currently enabled in configuration.
			@param[in] expression - single face expression
			@return true - enabled, false - disabled.
 		*/
		virtual pxcBool PXCAPI IsExpressionEnabled(PXCFaceData::ExpressionsData::FaceExpression expression) = 0;
	};

	class RecognitionConfiguration
	{
	public:
		enum RecognitionRegistrationMode
		{
			REGISTRATION_MODE_CONTINUOUS,	//registers users automatically
			REGISTRATION_MODE_ON_DEMAND,	//registers users on demand only
		};

		PXC_DEFINE_CONST(STORAGE_NAME_SIZE, 50);

		struct RecognitionStorageDesc
		{
			//pxcCHAR storageName[STORAGE_NAME_SIZE];
			pxcBool isPersistent;	//determines whether the database is saved on exit. Currently this value is ignored
			pxcI32 maxUsers;	//maximum number of people to keep in DB
			pxcI32 reserved[10];
		};
		RecognitionStorageDesc storageDesc;

		struct RecognitionProperties
		{
			pxcBool isEnabled;
			pxcI32 accuracyThreshold;
			RecognitionRegistrationMode registrationMode;
			pxcI32 reserved[10];
		};
		RecognitionProperties properties;

		__inline void Enable()
		{
			properties.isEnabled = true;
		}
		__inline void Disable()
		{
			properties.isEnabled = false;
		}
		__inline void SetAccuracyThreshold(pxcI32 threshold)
		{
			properties.accuracyThreshold = threshold;
		}
		__inline pxcI32 GetAccuracryThreshold()
		{
			return properties.accuracyThreshold;
		}
		__inline void SetRegistrationMode(RecognitionRegistrationMode mode)
		{
			properties.registrationMode = mode;
		}
		__inline RecognitionRegistrationMode GetRegistrationMode()
		{
			return properties.registrationMode;
		}
		/** 
			@brief Sets the active Recognition database.
			@param[in] storageName - The name of the database to be loaded by the Recognition module.
			@param[in] storage - A pointer to the Recognition database, or NULL for an existing database.
			@return PXC_STATUS_HANDLE_INVALID - if the module wasn't initialized properly.
			PXC_STATUS_DATA_UNAVAILABLE - if the registration failed.
			PXC_STATUS_NO_ERROR - if registration was successful.
		*/
		virtual pxcStatus PXCAPI UseStorage(pxcCHAR* storageName) = 0;

		/** 
			@brief Retrieves the Recognition storage descriptor for the active storage.
			@param[in] outStorage - The storage descriptor into which the active storage information will be copied.
		*/
		virtual pxcStatus PXCAPI QueryActiveStorage(RecognitionStorageDesc* outStorage) = 0;

		/** 
			@brief Create a new Recognition database.
			@param[in] storageName The name of the new database.
			@return Pointer to the new database.
		*/
		virtual pxcStatus PXCAPI CreateStorage(pxcCHAR* storageName, RecognitionStorageDesc* storageDesc) = 0;

		/** 
			@brief Sets a storage descriptor to an existing Recognition storage.
			@param[in] storageName - The name of the storage to set the descriptor for.
			@param[in] storageDesc - Holds the new descriptor for the selected storage.
			@return TBD
		*/
		virtual pxcStatus PXCAPI SetStorageDesc(pxcCHAR* storageName, RecognitionStorageDesc* storageDesc) = 0;

		/** 
			@brief Deletes an existing Recognition storage.
			@param[in] storageName The name of the storage to be deleted.
		*/
		virtual pxcStatus PXCAPI DeleteStorage(pxcCHAR* storageName) = 0;

		/** 
			@brief Sets an existing database as the Recognition database used by the Face Recognition module.
			@param[in] buffer The byte stream representing the Recognition database to be used.
		*/
		virtual void PXCAPI SetDatabaseBuffer(pxcBYTE* buffer, pxcI32 size) = 0;

	protected:
		virtual ~RecognitionConfiguration() {}
	};
	
	enum TrackingModeType
	{
		FACE_MODE_COLOR,
		FACE_MODE_COLOR_PLUS_DEPTH
	};

	DetectionConfiguration detection;
	LandmarksConfiguration landmarks;
	PoseConfiguration pose;
	virtual ExpressionsConfiguration* PXCAPI QueryExpressions() = 0;
	TrackingStrategyType strategy;

	virtual RecognitionConfiguration* QueryRecognition() = 0;

	virtual pxcStatus PXCAPI SetTrackingMode(TrackingModeType trackingMode) = 0;
	virtual TrackingModeType PXCAPI GetTrackingMode() = 0;
	
	/* -------------------------------------------------- ALERTS ------------------------------------------------- */
 	/*
 		@class AlertHandler
 		Interface for a callback for all categories of events 
 	*/
 	class AlertHandler
	{
 	public:
 		/*
 			@brief The OnFiredAlert method is called when a registered alert event is fired.
 			@param[in] alertData contains all the information for the fired event.
 			@see AlertData
 		*/
 		virtual void PXCAPI OnFiredAlert(const PXCFaceData::AlertData *alertData) = 0;
 	};

 	/*
 		@brief Enable alert, so that events are fired when the alert is identified.			
 		@param[in] alertEvent the label of the alert to enabled. 
 		@return PXC_STATUS_NO_ERROR if the alert was enabled successfully; otherwise, return one of the following errors:
 		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter.
 		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
 	*/
 	virtual pxcStatus PXCAPI EnableAlert(PXCFaceData::AlertData::AlertType alertEvent) = 0;
 
 	/*
 		@brief Enable all alert messaging events.
 		@return PXC_STATUS_NO_ERROR if enabling all alerts was successful; otherwise, return one of the following errors:
 		PXC_STATUS_PROCESS_FAILED - Module failure during processing.
 		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
 	*/
 	virtual void PXCAPI EnableAllAlerts(void) = 0;
         
 	/*
 		@brief Check if an alert is enabled.    
 		@param[in] alertEvent the ID of the event.            
 		@return true if the alert is enabled; otherwise, return false
 	*/
 	virtual pxcBool PXCAPI IsAlertEnabled(PXCFaceData::AlertData::AlertType alertEvent) const = 0;
         
 	/*
 		@brief Disable alert messaging for a specific event.            
 		@param[in] alertEvent the ID of the event to be disabled.
 		@return PXC_STATUS_NO_ERROR if disabling the alert was successful; otherwise, return one of the following errors:
 		PXC_STATUS_PARAM_UNSUPPORTED - Unsupported parameter.
 		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
 	*/
 	virtual pxcStatus PXCAPI DisableAlert(PXCFaceData::AlertData::AlertType alertEvent) = 0;
 
 	/*
 		@brief Disable all alerts messaging for all events.                        
 		@return PXC_STATUS_NO_ERROR if disabling all alerts was successful; otherwise, return one of the following errors:
 		PXC_STATUS_PROCESS_FAILED - Module failure during processing.
 		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
 	*/
 	virtual void PXCAPI DisableAllAlerts(void) = 0;        
                 
 	/*
 		@brief Register an event handler object for the alerts. The event handler's OnFiredAlert method will be called each time an alert is identified.
 		@param[in] alertHandler a pointer to the event handler.
 		@see AlertHandler::OnFiredAlert
 		@return PXC_STATUS_NO_ERROR if the registering an event handler was successful; otherwise, return the following error:
 		PXC_STATUS_DATA_NOT_INITIALIZED - Data failed to initialize.
 	*/
 	virtual pxcStatus PXCAPI SubscribeAlert(AlertHandler *alertHandler) = 0;
 
 	/*
 		@brief Unsubscribe an event handler object for the alerts.
 		@param[in] alertHandler a pointer to the event handler that should be removed.
 		@return PXC_STATUS_NO_ERROR if the unregistering the event handler was successful, an error otherwise.
 	*/
 	virtual pxcStatus PXCAPI UnsubscribeAlert(AlertHandler *alertHandler) = 0;

	/*
	* Commits configuration changes on module.
	*/
	virtual pxcStatus PXCAPI ApplyChanges() = 0;

	/*  
	* Restores configuration to global default values
	*/
	virtual void PXCAPI RestoreDefaults() = 0;

	/**
	* @brief Updates data to latest available configuration.
	*/
	virtual pxcStatus PXCAPI Update() = 0;
};