#pragma once

#include <cstdint> // For standard sized integer types
#include <climits> // For int limits

#ifdef DS_EXPORTS
#define DS_DECL \
    __declspec(dllexport)
#else
#define DS_DECL \
    __declspec(dllimport)
#endif

/// @defgroup Global Types and Constants
/// Defines DSAPI global data types, typedefs, and constants.
/// @{

/// Platform
enum DSPlatform
{
    DS_DS4_PLATFORM = 0,                                       ///< Use DS 4 device over a USB connection
    DS_DS5_PLATFORM = 1,                                       ///< Use DS 5 device over a USB connection
    DS_FILE_PLATFORM = 0x10000,                                ///< Use data previously recorded to disk
    DS_DS4_FILE_PLATFORM = DS_DS4_PLATFORM | DS_FILE_PLATFORM, ///< Use data previously recorded to disk from a DS4 device
    DS_DS5_FILE_PLATFORM = DS_DS5_PLATFORM | DS_FILE_PLATFORM  ///< Use data previously recorded to disk from a DS5 device
};

/// Z units
enum DSZUnits
{
    DS_MILLIMETERS = 1000,  ///< mm
    DS_CENTIMETERS = 10000, ///< cm
    DS_METERS = 1000000,    ///< m
    DS_INCHES = 25400,      ///< in
    DS_FEET = 304800        ///< ft
};

/// DSPixelFormat specifies one of the legal pixel formats.
enum DSPixelFormat
{
    DS_LUMINANCE8,            ///< Luminance 8 bits
    DS_LUMINANCE16,           ///< Luminance 16 bits
    DS_RGB8,                  ///< Color 8 bits color in order red, green, blue
    DS_BGRA8,                 ///< Color 8 bits color in order blue, green, red, alpha (Equivalent to D3DCOLOR, GL_BGRA)
    DS_NATIVE_L_LUMINANCE8,   ///< Native luminance format, equivalent to DS_LUMINANCE8, for left image only (right image must be disabled)
    DS_NATIVE_L_LUMINANCE16,  ///< Native luminance format, equivalent to DS_LUMINANCE16, for left image only (right image must be disabled)
    DS_NATIVE_RL_LUMINANCE8,  ///< Native pixel-interleaved stereo luminance format, 8 bit pixel from right image, then 8 bit pixel from left image (all 8 bits used)
    DS_NATIVE_RL_LUMINANCE12, ///< Native pixel-interleaved stereo luminance format, 12 bit pixel from right image, then 12 bit pixel from left image (call maxLRBits() to determine how many are used)
    DS_NATIVE_RL_LUMINANCE16, ///< Native pixel-interleaved stereo luminance format, 16 bit pixel from right image, then 16 bit pixel from left image (call maxLRBits() to determine how many are used)
    DS_NATIVE_YUY2,           ///< Native color format, 32 bits per 2x1 pixel patch, with separate 8-bit luminance values (Y), and shared 8-bit chrominance values (U,V) in Y0 U Y1 V order.
    DS_NATIVE_RAW10           ///< Native color format, 40 bits per 4x1 pixel patch, in an 8:8:8:8:2:2:2:2 pattern (10 bits per pixel, high bits first). Bayer patterned: Even rows represent R G R G, odd rows represent G B G B.
};
DS_DECL const char* DSPixelFormatString(DSPixelFormat pixelFormat);

/// Status information returned by many DSAPI low level functions.
enum DSStatus
{
    DS_NO_ERROR = 0,        ///< Call was successful.
    DS_FAILURE,             ///< Call failed.
    DS_INVALID_CALL,        ///< Can't call this function in the current "state" (before/after startCapture(), etc)
    DS_INVALID_ARG,         ///< The user passed an argument that is invalid for this function
    DS_INVALID_CALIBRATION, ///< Device has not been calibrated or has invalid calibration data
    DS_HARDWARE_MISSING,    ///< The device is not plugged in / not enumerating
    DS_HARDWARE_ERROR,      ///< Something bad happened on the hardware
    DS_HARDWARE_IN_USE,     ///< Hardware is being used by another application.
    DS_FILE_ERROR,          ///< Failed to find, read from, or write to file
    DS_MEMORY_ERROR,        ///< Ran out of memory
    DS_UNSUPPORTED_CONFIG   ///< Current configuration setting is unsupported, change your configuration
};
DS_DECL const char* DSStatusString(DSStatus status);

/// DSWhichImager specifies which imager (or both) is (are) being addressed
enum DSWhichImager
{
    DS_LEFT_IMAGER = 0x1,       ///< Address only the left imager
    DS_RIGHT_IMAGER = 0x2,      ///< Address only the right imager
    DS_BOTH_IMAGERS = 0x3,      ///< Address both imagers
    DS_THIRD_IMAGER = 0x10,     ///< Address Third Imager
    DS_ALL_THREE_IMAGERS = 0x13 ///< Address Left, Right and Third imager
};

/// DSWhichSensor specifies which sensor is being addressed
enum DSWhichSensor
{
    DS_TEMPERATURE_SENSOR_1 = 0x1000 ///< Address individual temperature sensors, if any are present
};

/// DSAPI chip types
enum DSChipType
{
    DS3_CHIP = 30,
    DS4_CHIP = 40,
    DS5_CHIP = 50
};

/// Power line frequence options
enum DSPowerLineFreqOption
{
    DS_POWER_LINE_FREQ_50 = 50,
    DS_POWER_LINE_FREQ_60 = 60
};

#if _MSC_VER < 1800 // Visual Studio versions prior to 2013 do not allow functions to be declared deleted
#define DS_DELETED_FUNCTION
#else
#define DS_DELETED_FUNCTION = delete
#endif

/// @}
