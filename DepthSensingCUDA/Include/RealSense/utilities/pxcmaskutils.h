/*******************************************************************************                                                                                                                                                                                                                          /*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file PXCMaskUtils.h
    Defines the PXCMaskUtils interface
 */
#pragma once
#include "pxcimage.h"

/** 
	@class PXCContourExtractor
	A utility for extracting contour lines from blob (mask) images.
	Given a mask image, in which the blob pixels are white (255) and the rest are black (0)
	this utility will extract the contour lines of the blob.
	The contour lines are all the lines that define the borders of the blob.
	Inner contour lines (i.e. "holes" in the blob) are defined by an array of clock-wise points.
	The outer contour line (i.e. the external border) is defined by an array of counter-clock-wise points.
*/
class PXCContourExtractor: public PXCBase {
public:

	PXC_CUID_OVERWRITE(0x98fc2453);

	/**
		@brief initialize PXCContourExtractor instance for a specific image type (size)
		@param[in] imageInfo definition of the images that should be processed
		@see PXCImage::ImageInfo
	*/
	virtual void PXCAPI Init(const PXCImage::ImageInfo& imageInfo) = 0;

	/**			
		@brief Extract the contour of the blob in the given image
		Given an image of a blob, in which object pixels are white (set to 255) and all other pixels are black (set to 0),
		extract the contour lines of the blob.
		Note that there might be multiple contour lines, if the blob contains "holes".
		@param[in] blobImage the blob-image to be processed
		@return PXC_STATUS_NO_ERROR if a valid blob image exists and could be processed; otherwise, return the following error:
		PXC_STATUS_DATA_UNAVAILABLE - if image is not available or PXC_STATUS_ITEM_UNAVAILABLE if processImage is running or PXCContourExtractor was not initialized.		
	*/		
	virtual pxcStatus PXCAPI ProcessImage(const PXCImage::ImageData& blobImage) = 0; 

	/** 
		@brief Get the data of the contour line
		A contour is composed of a line, an array of 2D points describing the contour path
		@param[in] index the zero-based index of the requested contour
		@param[in] maxSize size of the allocated array for the contour points
		@param[out] contour points stored in the user allocated array		
		@return PXC_STATUS_NO_ERROR if terminated successfully; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED if index is invalid or user allocated array is invalid 		
		PXC_STATUS_ITEM_UNAVAILABLE if processImage is running or PXCContourExtractor was not initialized.
	*/
	virtual pxcStatus PXCAPI QueryContourData(const pxcI32 index, const pxcI32 maxSize, PXCPointI32* contour) = 0;

	/** 
		@brief Return true if it is the blob's outer contour, false for internal contours.
		@return true if it is the blob's outer contour, false for internal contours.
		@param[in] index the zero-based index of the requested contour
	*/
	virtual pxcBool PXCAPI IsContourOuter(const pxcI32 index) const = 0;

	/** 
		@brief Get the contour size (number of points in the contour)
		This is the size of the points array that the user should allocate
		@return the contour size (number of points in the contour)
		@param[in] index the zero-based index of the requested contour
	*/
	virtual pxcI32 PXCAPI QueryContourSize(const pxcI32 index) const = 0;	

	/** 
		@brief Get the number of contours extracted
		@return the number of contours extracted
	*/
	virtual pxcI32 PXCAPI QueryNumberOfContours(void) const = 0;
	
	/** 
		@brief Set the smoothing level of the shape of the contour 
		The smoothing level ranges from 0 to 1, when 0 means no smoothing, and 1 implies a very smooth contour
		Note that a larger smoothing level will reduce the number of points, while "cleaning" small holes in the line
		@param[in] smoothing the smoothing level
		@return PXC_STATUS_NO_ERROR if smoothing is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, smoothing level will remain the last valid value
	*/
	virtual pxcStatus PXCAPI SetSmoothing(pxcF32 smoothing) = 0;

	/** 
		@brief Get the smoothing level of the contour (0-1) 
		@return smoothing level of the contour
	*/
	virtual pxcF32 PXCAPI QuerySmoothing(void) const = 0;	
	
};

/**
	@class PXCBlobExtractor
	A utility for extracting mask images (blobs) of objects in front of the camera.
	Given a depth image, this utility will find the largest objects that are "close enough" to the camera.
	For each object a segmentation mask image will be created, that is, all object pixels will be "white" (255) 
	and all other pixels will be "black" (0).
	The maximal number of blobs that can be extracted is 2.
	The order of the blobs will be from the largest to the smallest (in number of pixels)
*/
class PXCBlobExtractor: public PXCBase {
public:

	PXC_CUID_OVERWRITE(0xa52305bc);

	/** 
	    @struct BlobData
		Contains the parameters that define a blob
    */
	struct BlobData {	
		PXCPointI32		closestPoint;		/// Image coordinates of the closest point in the blob
		PXCPointI32		leftPoint;			/// Image coordinates of the left-most point in the blob
		PXCPointI32		rightPoint;			/// Image coordinates of the right-most point in the blob
		PXCPointI32		topPoint;			/// Image coordinates of the top point in the blob
		PXCPointI32		bottomPoint;		/// Image coordinates of the bottom point in the blob
		PXCPointF32		centerPoint;		/// Image coordinates of the center of the blob
		pxcI32			pixelCount;			/// The number of pixels in the blob
	};

	/**
		@brief initialize PXCBlobExtractor instance for a specific image type (size)
		@param[in] imageInfo definition of the images that should be processed
		@see PXCImage::ImageInfo
	*/
	virtual void PXCAPI Init(const PXCImage::ImageInfo& imageInfo) = 0;

	/**			
		@brief Extract the 2D image mask of the blob in front of the camera. 	 
		In the image mask, each pixel occupied by the blob's is white (set to 255) and all other pixels are black (set to 0).
		@param[in] depthImage the depth image to be segmented		
		@return PXC_STATUS_NO_ERROR if a valid depth exists and could be segmented; otherwise, return the following error:
		PXC_STATUS_DATA_UNAVAILABLE - if image is not available or PXC_STATUS_ITEM_UNAVAILABLE if processImage is running or PXCBlobExtractor was not initialized.		
	*/		
	virtual pxcStatus PXCAPI ProcessImage(const PXCImage::ImageData& depthImage) = 0; 

	/**
		@brief Retrieve the 2D image mask of the blob and its associated blob data
		The blobs are ordered from the largest to the smallest (in number of pixels)
		@see BlobData
		@param[in] index the zero-based index of the requested blob (has to be between 0 to number of blobs) 
		@param[out] segmentationImage the 2D image mask of the requested blob
		@param[out] blobData the data of the requested blob
		@return PXC_STATUS_NO_ERROR if index is valid and processImage terminated successfully; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED if index is invalid or PXC_STATUS_ITEM_UNAVAILABLE if processImage is running or PXCBlobExtractor was not initialized.
	*/
	virtual pxcStatus PXCAPI QueryBlobData(const pxcI32 index, PXCImage::ImageData& segmentationImage, BlobData& blobData) = 0;
		
	/** 
		@brief Set the maximal number of blobs that can be detected 
		The default number of blobs that will be detected is 1
		@param[in] maxBlobs the maximal number of blobs that can be detected (limited to 2)
		@return PXC_STATUS_NO_ERROR if maxBlobs is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, maxBlobs will remain the last valid value
	
	*/
	virtual pxcStatus PXCAPI SetMaxBlobs(const pxcI32 maxBlobs) = 0;

	/**
		@brief Get the maximal number of blobs that can be detected 	
		@return the maximal number of blobs that can be detected 	
	*/
	virtual pxcI32 PXCAPI QueryMaxBlobs(void) const = 0;

	/** 
		@brief Get the number of detected blobs  	
		@return the number of detected blobs  	
	*/
	virtual pxcI32 PXCAPI QueryNumberOfBlobs(void) const = 0;

	/** 
		@brief Set the maximal distance limit from the camera. 
		Blobs will be objects that appear between the camera and the maxDistance limit.
		@param[in] maxDistance the maximal distance from the camera (has to be a positive value) 
		@return PXC_STATUS_NO_ERROR if maxDistance is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, maxDistance will remain the last valid value
	*/
	virtual pxcStatus PXCAPI SetMaxDistance(pxcF32 maxDistance) = 0;

	/** 
		@brief Get the maximal distance from the camera, in which an object can be detected and segmented
		@return maximal distance from the camera
	*/
	virtual pxcF32 PXCAPI QueryMaxDistance(void) const = 0;
	
	/**
		@brief Set the maximal depth of a blob (maximal distance between closest and furthest points on blob)
		@param[in] maxDepth the maximal depth of the blob (has to be a positive value) 
		@return PXC_STATUS_NO_ERROR if maxDepth is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, maxDepth will remain the last valid value
	*/
	virtual pxcStatus PXCAPI SetMaxObjectDepth(pxcF32 maxDepth) = 0;

	/** 
		@brief Get the maximal depth of the blob that can be detected and segmented
		@return maximal depth of the blob
	*/

	virtual pxcF32 PXCAPI QueryMaxObjectDepth(void) const = 0;	

	/** 
		@brief Set the smoothing level of the shape of the blob
		The smoothing level ranges from 0 to 1, when 0 means no smoothing, and 1 implies a very smooth contour
		@param[in] smoothing the smoothing level
		@return PXC_STATUS_NO_ERROR if smoothing is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, smoothing level will remain the last valid value
	*/
	virtual pxcStatus PXCAPI SetSmoothing(pxcF32 smoothing) = 0;

	/** 
		@brief Get the smoothing level of the blob (0-1) 
		@return smoothing level of the blob
	*/
	virtual pxcF32 PXCAPI QuerySmoothing(void) const = 0;	

	/** 
		@brief Set the minimal blob size in pixels
		Any blob that is smaller than threshold will be cleared during "ProcessImage".
		@param[in] minBlobSize the minimal blob size in pixels (cannot be more than a quarter of image-size)
		@return PXC_STATUS_NO_ERROR if minBlobSize is valid; otherwise, return the following error:
		PXC_STATUS_PARAM_UNSUPPORTED, minimal blob size will remain the last valid size
	*/
	virtual pxcStatus PXCAPI SetClearMinBlobSize(pxcI32 minBlobSize) = 0;

	/** 
		@brief Get the minimal blob size in pixels
		@return minimal blob size
	*/
	virtual pxcI32 PXCAPI QueryClearMinBlobSize(void) const = 0;	
};
