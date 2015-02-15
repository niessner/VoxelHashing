#pragma once

#include <cstdint>

const int DS_MAX_NUM_INTRINSICS_RIGHT = 2;      ///< Max number right cameras supported (e.g. one or two, two would support a multi-baseline unit)
const int DS_MAX_NUM_INTRINSICS_THIRD = 3;      ///< Max number native resolutions the third camera can have (e.g. 1920x1080 and 640x480)
const int DS_MAX_NUM_RECTIFIED_MODES_LR = 4;    ///< Max number rectified LR resolution modes the structure supports (e.g. 640x480, 492x372 and 332x252)
const int DS_MAX_NUM_RECTIFIED_MODES_THIRD = 4; ///< Max number rectified Third resolution modes the structure supports (e.g. 1920x1080, 1280x720, 640x480 and 320x240)

struct DSCalibIntrinsicsNonRectified
{
    /// A 3D point X = (x,y,z) in the non-rectified camera coordinate system maps to a non-rectified image pixel (u,v) by:
    ///
    /// u' = x / z
    /// v' = y / z
    /// r = sqrt (u'^2 + v'^2)
    ///
    /// u'' = u' * (1 + k[0]*r^2 + k[1]*r^4 + k[4]*r^6)
    /// v'' = v' * (1 + k[0]*r^2 + k[1]*r^4 + k[4]*r^6)
    ///
    /// u''' =  u'' + 2 * k[2] * u'' * v'' + k[3]*(r^2 + 2 * u''^2)
    /// v''' =  v'' + 2 * k[3] * u'' * v'' + k[2]*(r^2 + 2 * v''^2)
    ///
    /// u = fx * u''' + px
    /// v = fy * v''' + py
    float fx;
    float fy;
    float px;
    float py;
    double k[5];
    uint32_t w;
    uint32_t h;
};

struct DSCalibIntrinsicsRectified
{
    /// A 3D point X' = (x',y',z') in the rectified camera coordinate system maps to a rectified image pixel (u,v) by:
    ///
    /// u' = x' / z'
    /// v' = y' / z'
    ///
    /// u = rfx * u' + rpx
    /// v = rfy * v' + rpy
    ///
    /// Rectified camera coordinates X' are related to non-rectified camera coordinates X by a rotation matrix. See below.
    float rfx;
    float rfy;
    float rpx;
    float rpy;
    uint32_t rw;
    uint32_t rh;
};

struct DSCalibRectParameters
{
    /// This structure represents calibration and rectification information for a DS module comprising at least one left
    /// camera LC and a right camera RC. There can be an extra right camera RC' to support multi-baseline
    /// configurations. There can be a third camera (oftentimes a higher-resolution color camera); this camera may natively
    /// support two resolutions TC or TC'. Each camera/resolution LC, RC, RC', TC, TC' has its own intrinsics
    /// (DSCalibIntrinsicsNonRectified).
    ///
    /// This structure supports parameters defining rectified left and right images suitable for the DS stereo algorithm
    /// and rectification of third images such that they share the same image plane as the selected left and right images.
    ///
    /// The image plane of the rectified images depends uniquely on the left and right camera intrinsics and
    /// extrinsics. The image plane is encoded by the rotation matrices between the rectified camera coordinates X' =
    /// (x',y',z') and the non-rectified camera coordinates X = (x,y,z): X' = Rotation * X.
    ///
    ///      Xleft =  Rleft[indexIntrinsicsRight]  * X'left                        (rec-to-non-rec-coordinate-systems)
    ///      Xright = Rright[indexIntrinsicsRight] * X'right
    ///      Xthird = Rthird[indexIntrinsicsRight] * X'third
    ///
    /// where indexIntrinsicsRight is the index of the used right camera (e.g. RC or RC').
    ///
    /// Within an image plane, the field of view and the size of the rectified images is defined by the focal lengths,
    /// principal point and image size in a DSCalibIntrinsicsRectified. For each image plane, there are numRectifiedModesLR rectified 
    /// resolution modes for the left and right cameras and numRectifiedModesThird rectified resolution modes for the third camera.
    ///
    /// For a fixed image plane, the rectified camera coordinates are related by a translation (in millimeters):
    ///
    ///     X'right = X'left + [B 0 0]                                                     (rec-camera-to-rec-camera)
    ///     X'third = X'left + T
    ///
    /// Additionally, the module can be located wrt a "world" coordinate system. Non-rectified left to the world:
    ///
    ///     Xworld = Rworld * Xleft + Tworld
    ///
    /// Rotations are stored in row major format.
    ///
    /// SYNOPSIS:
    ///
    /// An operating mode of a DS module is defined by:
    ///
    /// - A struct c of type DSCalibRectParameters, and a selection of:
    /// - indexIntrinsicsRight, the index of the right camera (must be < c.numIntrinsicsRight)
    /// - indexIntrinsicsThird, the index of the native resolution of the third camera (must be < c.numIntrinsicsThird)
    /// - indexRectifiedModeLR, the index of the rectified resolution mode of the left and right cameras (must be < c.numRectifiedModesLR)
    /// - indexRectifiedModeThird, the index of the rectified resolution mode of the third camera (must be < c.numRectifiedModesThird)
    ///
    /// For a given operating mode,
    ///
    ///     The intrinsics of the left  camera are in c.intrinsicsLeft
    ///     The intrinsics of the right camera are in c.intrinsicsRight[indexIntrinsicsRight]
    ///     The intrinsics of the third camera are in c.intrinsicsThird[indexIntrinsicsThird]
    ///
    ///     The rectified-to-non-rectified rotations are defined in c.Rleft[indexIntrinsicsRight], c.Rright[indexIntrinsicsRight] and
    ///     c.Rthird[indexIntrinsicsRight] (as in Eq. (rec-to-non-rec-coordinate-systems)).
    ///
    ///     The translations between rectified camera coordinates are defined by c.B[indexIntrinsicsRight] and c.T[indexIntrinsicsRight]
    ///     (as in Eq. (rec-camera-to-rec-camera)).
    ///
    ///     The field of view and sizes of the rectified images are defined in c.modesLR[indexIntrinsicsRight][indexRectifiedModeLR] and
    ///     c.modesThird[indexIntrinsicsRight][indexRectifiedModeThird].
    ///
    /// NOMENCLATURE:
    ///
    /// Non-rectified camera coordinates: 3D coordinates where the X- and Y-axes are aligned with the imager's pixel rows
    ///     and columns, respectively. The Z-axis points forward, from the camera to the scene. This forms a right-handed
    ///     coordinate system.
    ///
    /// Rectified camera coordinates: 3D coordinates where the X- and Y-axes are aligned with the rectified image plane's
    ///     x- and y-axes. The Z-axis points forward (a right-handed coordinate system). The left and right images will be
    ///     aligned in Y such that a scene point will be on the same row in both images. The origin of e.g.  the left
    ///     rectified coordinate system is the center of projection of that camera. The origin is the same as that of the
    ///     non-rectified coordinate system; they differ by a rotation (typically a few degrees at most).  The rotation is such
    ///     that the X-axis points straight from the left camera's center of projection to the center of projection of the
    ///     right camera.
    ///
    /// Rectified image plane: The plane parallel to the rectified camera coordinate's X and Y axes.

    uint32_t versionNumber; ///< Version of this format

    uint16_t numIntrinsicsRight;     ///< Number of right cameras < DS_MAX_NUM_INTRINSICS_RIGHT
    uint16_t numIntrinsicsThird;     ///< Number of native resolutions of third camera < DS_MAX_NUM_INTRINSICS_THIRD
    uint16_t numRectifiedModesLR;    ///< Number of rectified LR resolution modes < DS_MAX_NUM_RECTIFIED_MODES_LR
    uint16_t numRectifiedModesThird; ///< Number of rectified Third resolution modes < DS_MAX_NUM_RECTIFIED_MODES_THIRD

    DSCalibIntrinsicsNonRectified intrinsicsLeft;
    DSCalibIntrinsicsNonRectified intrinsicsRight[DS_MAX_NUM_INTRINSICS_RIGHT];
    DSCalibIntrinsicsNonRectified intrinsicsThird[DS_MAX_NUM_INTRINSICS_THIRD];

    DSCalibIntrinsicsRectified modesLR[DS_MAX_NUM_INTRINSICS_RIGHT][DS_MAX_NUM_RECTIFIED_MODES_LR];
    DSCalibIntrinsicsRectified modesThird[DS_MAX_NUM_INTRINSICS_RIGHT][DS_MAX_NUM_INTRINSICS_THIRD][DS_MAX_NUM_RECTIFIED_MODES_THIRD];

    double Rleft[DS_MAX_NUM_INTRINSICS_RIGHT][9];
    double Rright[DS_MAX_NUM_INTRINSICS_RIGHT][9];
    double Rthird[DS_MAX_NUM_INTRINSICS_RIGHT][9];

    float B[DS_MAX_NUM_INTRINSICS_RIGHT];
    float T[DS_MAX_NUM_INTRINSICS_RIGHT][3];

    double Rworld[9];
    float Tworld[3];
};
