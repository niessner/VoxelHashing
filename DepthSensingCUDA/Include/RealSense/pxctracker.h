/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/

/** @file pxctracker.h
Defines the PXCTracker interface, which programs may use for 3D tracking.
*/
#pragma once
#include "pxcsession.h"
#include "pxccapture.h"
#pragma warning(push)
#pragma warning(disable:4201) /* nameless structs/unions */

/**
This class defines a standard interface for 3D tracking algorithms. 
*/
class PXCTracker:public PXCBase {
public:
	PXC_CUID_OVERWRITE(PXC_UID('T','R','K','R'));

	/**
	* The tracking states of a target.
	*
	* The state of a target usually starts with ETS_NOT_TRACKING.
	* When it is found in the current camera image, the state change to
	* ETS_FOUND for one image, the following images where the location of the
	* target is successfully determined will have the state ETS_TRACKING.
	*
	* Once the tracking is lost, there will be one single frame ETS_LOST, then
	* the state will be ETS_NOT_TRACKING again. In case there is extrapolation 
	* of the pose requested, the transition may be from ETS_TRACKING to ETS_EXTRAPOLATED.
	*
	* To sum up, these are the state transitions to be expected:
	*  ETS_NOT_TRACKING -> ETS_FOUND 
	*  ETS_FOUND        -> ETS_TRACKING
	*  ETS_TRACKING     -> ETS_LOST
	*  ETS_LOST         -> ETS_NOT_TRACKING
	*
	* With additional extrapolation, these transitions can occur as well:
	*  ETS_TRACKING     -> ETS_EXTRAPOLATED
	*  ETS_EXTRAPOLATED -> ETS_LOST
	*
	* "Event-States" do not necessarily correspond to a complete frame but can be used to 
	* flag individual tracking events or replace tracking states to clarify their context:
	*  ETS_NOT_TRACKING -> ETS_REGISTERED -> ETS_FOUND for edge based initialization
	*/
	enum ETrackingState
	{
		ETS_UNKNOWN		 = 0,	///< Tracking state is unknown
		ETS_NOT_TRACKING = 1,	///< Not tracking
		ETS_TRACKING	 = 2,	///< Tracking
		ETS_LOST		 = 3,	///< Target lost
		ETS_FOUND		 = 4,	///< Target found
		ETS_EXTRAPOLATED = 5,	///< Tracking by extrapolating
		ETS_INITIALIZED	 = 6,	///< The tracking has just been loaded

		ETS_REGISTERED	 = 7	///< Event-State: Pose was just registered for tracking
	};

	struct TrackingValues {
		ETrackingState			state;			///< The state of the tracking values

		PXCPoint3DF32           translation;	///< Translation component of the pose
		PXCPoint4DF32           rotation;		///< Rotation component of the pose

		/** 
		* Quality of the tracking values.
		* Value between 0 and 1 defining the tracking quality.
		* A higher value means better tracking results. More specifically:
		* - 1 means the system is tracking perfectly.
		* - 0 means that we are not tracking at all.
		*/
		pxcF32				quality;

		pxcF64				timeElapsed;				///< Time elapsed (in ms) since last state change of the tracking system
		pxcF64				trackingTimeMs;				///< Time (in milliseconds) used for tracking the respective frame
		pxcI32				cosID;						///< The ID of the coordinate system
		pxcCHAR				targetName[256];            ///< The name of the target object
		pxcCHAR				additionalValues[256];      ///< Extra space for information provided by a sensor that cannot be expressed with translation and rotation properly.
		pxcCHAR				sensor[256];                ///< The sensor that provided the values

		pxcI32				reserved[32];               // 0 - reserved for module specific parameters
	};


	/// Set the camera parameters, which can be the result of camera calibration from the toolbox
	virtual pxcStatus PXCAPI SetCameraParameters(const pxcCHAR *filename)=0;


	/**
	* Add a 2D reference image for tracking a target

	* \param filename: path to image file
	* \param[out] cosID: coordinate system ID of added target
	* \param widthMM: image width in mm (optional)
	* \param heightMM: image height in mm (optional)
	* \param qualityThreshold: Reserved								
	*/
	virtual pxcStatus PXCAPI Set2DTrackFromFile(const pxcCHAR *filename, pxcUID& cosID, pxcF32 widthMM, pxcF32 heightMM, pxcF32 qualityThreshold) = 0;
	pxcStatus __inline Set2DTrackFromFile(const pxcCHAR *filename, pxcUID& cosID) { return Set2DTrackFromFile(filename, cosID, 0.f, 0.f, 0.7f); }

	virtual pxcStatus PXCAPI Set2DTrackFromImage(PXCImage *image, pxcUID& cosID, pxcF32 widthMM, pxcF32 heightMM, pxcF32 qualityThreshold) = 0;
	pxcStatus __inline Set2DTrackFromImage(PXCImage *image, pxcUID& cosID) { return Set2DTrackFromImage(image, cosID, 0.f, 0.f, 0.7f); }

	/**
	* Add a 3D tracking configuration for a target
	*
	* This file can be generated with the Toolbox
	*
	* \param filename The full path to the configuration file (*.slam, *.xml)
	* \param[out] firstCosID: coordinate system ID of the first added target
	* \param[out] lastCosID: coordinate system ID of the last added target (may be the same as \c firstCosID)
	*/
	virtual pxcStatus PXCAPI Set3DTrack(const pxcCHAR *filename, pxcUID& firstCosID, pxcUID& lastCosID) = 0;

	/**
	* Enable instant 3D tracking (SLAM).  This form of tracking does not require a target model to be
	* previously created and loaded.
	*
	* \param egoMotion: Specify the world coordinate system origin of the tracked objects.
						\c true uses the first image captured from the camera
						\c false (default) uses the "main plane" of the scene which is determined heuristically	
	*/
	virtual pxcStatus PXCAPI Set3DInstantTrack(pxcBool egoMotion)=0;
	pxcStatus __inline Set3DInstantTrack(void) { return Set3DInstantTrack(false); }

	/**
	* Get the number of targets currently tracking
	* \return The number of active tracking targets
	*
	* \sa QueryTrackingValues, QueryAllTrackingValues
	*/
	virtual pxcI32 PXCAPI QueryNumberTrackingValues() const = 0;

	/**
	* Get information for all of the active tracking targets
	* 
	* \param trackingValues: Pointer to store the tracking results at.  The passed in block must be
	*						 at least QueryNumberTrackingValues() elements long
	*/
	virtual pxcStatus PXCAPI QueryAllTrackingValues(PXCTracker::TrackingValues *trackingValues)=0;

	/**
	* Return information for a particular coordinate system ID.  This value can be returned from Set2DTrackFromFile(),
	* Set2DTrackFromImage(), or Set3DTrack().  coordinate system IDs for Set3DInstantTrack() are generated dynamically as
	* targets that are determined in the scene.
	*
	* \param cosID: The coordinate system ID to return the status for
	* \param outTrackingValues: The returned tracking values. the user needs to manage the mapping between the cosIDs and targets in loaded.
	*/
	virtual pxcStatus PXCAPI QueryTrackingValues(pxcUID cosID, TrackingValues& outTrackingValues)=0;
};
