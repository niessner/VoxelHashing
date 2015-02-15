/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012-2014 Intel Corporation. All Rights Reserved.

*******************************************************************************/
/** @file pxcprojection.h
    Defines the PXCProjection interface, which defines mappings between
    pixel, depth, and real world coordinates.
  */
#pragma once
#include "pxcimage.h"

/**
    This interface defines mappings between various coordinate systems
    used by modules of the SDK. Call the PXCCapture::Device::CreateProjection 
    to create an instance of this interface.

    The class extends PXCSerializeableService. Use QueryInstance<PXCSerializeableService> 
    to access the PXCSerializableService interface.
 */
class PXCProjection: public PXCBase {
public:
    PXC_CUID_OVERWRITE(0x494A8537);

    /** 
        @brief Map depth coordinates to color coordinates for a few pixels.
        @param[in]  npoints         The number of pixels to be mapped.
        @param[in]  pos_uvz         The array of depth coordinates + depth value in the PXCPoint3DF32 structure.
        @param[out] pos_ij          The array of color coordinates, to be returned.
        @return PXC_STATUS_NO_ERROR Successful execution.
    */ 
    virtual pxcStatus PXCAPI MapDepthToColor(pxcI32 npoints, PXCPoint3DF32 *pos_uvz, PXCPointF32  *pos_ij)=0;

    /** 
        @brief Map color coordinates to depth coordiantes for a few pixels.
        @param[in]  depth           The depthmap image.
        @param[in]  npoints         The number of pixels to be mapped.
        @param[in]  pos_ij          The array of color coordinates.
        @param[out] pos_uv          The array of depth coordinates, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI MapColorToDepth(PXCImage *depth, pxcI32 npoints, PXCPointF32 *pos_ij, PXCPointF32 *pos_uv)=0;

    /** 
        @brief Map depth coordinates to world coordinates for a few pixels.
        @param[in]   npoints        The number of pixels to be mapped.
        @param[in]   pos_uvz        The array of depth coordinates + depth value in the PXCPoint3DF32 structure.
        @param[out]  pos3d          The array of world coordinates, in mm, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI ProjectDepthToCamera(pxcI32 npoints, PXCPoint3DF32 *pos_uvz, PXCPoint3DF32 *pos3d)=0;

    /** 
        @brief Map color pixel coordinates to camera coordinates for a few pixels.
        @param[in]   npoints        The number of pixels to be mapped.
        @param[in]   pos_ijz        The array of color coordinates + depth value in the PXCPoint3DF32 structure.
        @param[out]  pos3d          The array of camera coordinates, in mm, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI ProjectColorToCamera(pxcI32 npoints, PXCPoint3DF32 *pos_ijz, PXCPoint3DF32 *pos3d)=0;

    /** 
        @brief Map camera coordinates to depth coordinates for a few pixels.
        @param[in]    npoints       The number of pixels to be mapped.
        @param[in]    pos3d         The array of world coordinates, in mm.
        @param[out]   pos_uv        The array of depth coordinates, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI ProjectCameraToDepth(pxcI32 npoints, PXCPoint3DF32 *pos3d, PXCPointF32 *pos_uv)=0;

    /** 
        @brief Map camera coordinates to color coordinates for a few pixels.
        @param[in]    npoints       The number of pixels to be mapped.
        @param[in]    pos3d         The array of world coordinates, in mm.
        @param[out]   pos_ij        The array of color coordinates, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI ProjectCameraToColor(pxcI32 npoints, PXCPoint3DF32 *pos3d, PXCPointF32 *pos_ij)=0;

    /** 
        @brief Retrieve the UV map for the specific depth image. The UVMap is a PXCPointF32 array of depth size width*height.
        @param[in]  depth        The depth image instance.
        @param[out] uvmap        The UV map, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI QueryUVMap(PXCImage *depth, PXCPointF32 *uvmap)=0;

    /** 
        @brief Retrieve the inverse UV map for the specific depth image. The inverse UV map maps color coordinates
        back to the depth coordinates. The inverse UVMap is a PXCPointF32 array of color size width*height.
        @param[in]  depth        The depth image instance.
        @param[out] inv_uvmap    The inverse UV map, to be returned.
        @return PXC_STATUS_NO_ERROR     Successful execution.
    */ 
    virtual pxcStatus PXCAPI QueryInvUVMap(PXCImage *depth, PXCPointF32 *inv_uvmap)=0;

    /** 
        @brief Retrieve the vertices for the specific depth image. The vertices is a PXCPoint3DF32 array of depth 
        size width*height. The world coordiantes units are in mm.
        @param[in]  depth        The depth image instance.
        @param[out] inv_uvmap    The inverse UV map, to be returned.
        @return PXC_STATUS_NO_ERROR Successful execution.
    */ 
    virtual pxcStatus PXCAPI QueryVertices(PXCImage *depth, PXCPoint3DF32 *vertices)=0;

    /** 
        @brief Get the color pixel for every depth pixel using the UV map, and output a color image, aligned in space 
        and resolution to the depth image.
        @param[in] depth        The depth image instance.
        @param[in] color        The color image instance.
        @return The output image in the depth image resolution.
    */ 
    virtual PXCImage* PXCAPI CreateColorImageMappedToDepth(PXCImage *depth, PXCImage *color)=0;                 

    /** 
        @brief Map every depth pixel to the color image resolution using the UV map, and output an incomplete 
        depth image (with holes), aligned in space and resolution to the color image. 
        @param[in] depth        The depth image instance.
        @param[in] color        The color image instance.
        @return The output image in the color image resolution.
    */ 
    virtual PXCImage* PXCAPI CreateDepthImageMappedToColor(PXCImage *depth, PXCImage *color)=0;          
};
