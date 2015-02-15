#pragma once

#include "DSAPI/DSAPITypes.h"

/// @class DSHardware
/// Defines methods specific to a hardware implementation.
class DS_DECL DSHardware
{
public:
    /// Returns true if the hardware is active/ready.
    virtual bool isHardwareActive() = 0;

    /// Returns the version of DS ASIC used to compute data.
    virtual DSChipType getChipVersion() = 0;

    /// Not validated.
    /// Sets the value to subtract when estimating the median of the correlation surface.
    /// RobbinsMunroe is an incremental algorithm for approximating the median of set of numbers.  It works by maintaining a running
    /// estimate of the median.  If an incoming value is smaller than the estimate, the estimate is reduced by a certain amount, the
    /// minusIncrement.  The estimated median is used to evaluate the distinctiveness of the winning disparity.
    /// @param minusIncrement the increment to subtract when the new value is lower than the current estimate.
    /// @return false on fail else true.
    virtual bool setRobbinsMunroeMinusIncrement(uint32_t minusIncrement) = 0;
    /// Gets the value to subtract when estimating the median of the correlation surface.
    /// @return false on fail else true.
    virtual bool getRobbinsMunroeMinusIncrement(uint32_t& minusIncrement) = 0;

    /// Not validated.
    /// Sets the value to add when estimating the median of the correlation surface.
    /// RobbinsMunroe is an incremental algorithm for approximating the median of set of numbers.  It works by maintaining a running
    /// estimate of the median.  If an incoming value is greater than the estimate, the estimate is increased by a certain amount, the
    /// plusIncrement.  The estimated median is used to evaluate the distinctiveness of the winning disparity.
    /// @param plusIncrement the increment to add when the new value is greater than the current estimate.
    /// @return false on fail else true.
    virtual bool setRobbinsMunroePlusIncrement(uint32_t plusIncrement) = 0;
    /// Gets the value to add when estimating the median of the correlation surface.
    /// @return false on fail else true.
    virtual bool getRobbinsMunroePlusIncrement(uint32_t& plusIncrement) = 0;

    /// Not validated.
    /// Sets a threshold by how much the winning score must beat the median.
    /// RobbinsMunroe is an incremental algorithm for approximating the median of set of numbers.  It works by maintaining a running
    /// estimate of the median.  After the whole disparity search has been performed for a pixel, the result is evaluated based upon
    /// the distinctiveness of its correlation score. The score must be a threshold less than the median.
    /// @param threshold for a depth estimate to be considered valid, its correlation score must be a threshold less than the median.
    /// @return false on fail else true.
    virtual bool setMedianThreshold(uint32_t threshold) = 0;
    /// Gets a threshold by how much the winning score must beat the median.
    /// @return false on fail else true.
    virtual bool getMedianThreshold(uint32_t& threshold) = 0;

    /// Not validated.
    /// Sets the minimum and maximum correlation score that is considered acceptable.
    /// @param min if the correlation score for a pixel is less than min, it is marked invalid.
    /// @param max if the correlation score is greater than max, the pixel is marked invalid.
    /// @return false on fail else true.
    virtual bool setMinMaxScoreThreshold(uint32_t min, uint32_t max) = 0;
    /// Gets the minimum and maximum correlation score that is considered acceptable.
    /// @return false on fail else true.
    virtual bool getMinMaxScoreThreshold(uint32_t& min, uint32_t& max) = 0;

    /// Not validated.
    /// Sets the threshold on how much at least one adjacent disparity score must differ from the minimum score.
    /// @param val smallest correlation score must differ from one adjacent disparity by at least a val.
    /// @return false on fail else true.
    virtual bool setNeighborThreshold(uint32_t val) = 0;
    /// Gets a threshold on how much at least one adjacent disparity score must differ from the minimum score.
    /// @return false on fail else true.
    virtual bool getNeighborThreshold(uint32_t& val) = 0;

    /// Not validated.
    /// Determines the current threshold for determining whether the left-right match agrees with the right-left match.
    /// A traditional means of determining whether the disparity estimate for a pixel is correct is the so-called
    /// left-right check. The left-right disparity is computed with 5 bits of sub-pixel information.  The right-left match is
    /// computed to the nearest integer disparity. Clearly, in general, we cannot expect better than one half pixel difference
    /// between the two. The units of this value are 32nds of a pixel.  Thus a value of 16 would indicate a half-pixel threshold.
    /// @param val left-right and right-matches can differ by at most val/32
    /// @return false on fail else true.
    virtual bool setLRAgreeThreshold(uint32_t val) = 0;
    /// Gets the current threshold for determining whether the left-right match agrees with the right-left match.
    /// @return false on fail else true.
    virtual bool getLRAgreeThreshold(uint32_t& val) = 0;

    /// Not validated.
    /// Set parameter for determining whether the texture in the region is sufficient to justify a depth result.
    /// Some minimum level of texture is required in the region of a pixel to determine stereo correspondence.
    /// The textureCountThresh specifies how many neighboring pixels in a 7x7 area surrounding the current pixel 
    /// must meet the threshold on their difference in intensity from the current pixel.
    /// @return false on fail else true.
    virtual bool setTextureCountThreshold(uint32_t textureCountThresh) = 0;
    /// Get parameter for determining whether the texture in the region is sufficient to justify a depth result.
    /// @return false on fail else true.
    virtual bool getTextureCountThreshold(uint32_t& textureCountThresh) = 0;

    /// Not validated.
    /// Set parameter for determining whether the texture in the region is sufficient to justify a depth result.
    /// Some minimum level of texture is required in the region of a pixel to determine stereo correspondence.
    /// The textureDifferenceThresh specifies how much a neighboring pixel must differ from its neighbors to be deemed countable texture.
    /// @param textureDifferenceThresh the difference required to pass as texture.
    /// @return false on fail else true.
    virtual bool setTextureDifferenceThreshold(uint32_t textureDifferenceThresh) = 0;
    /// Get parameters for determining whether the texture in the region is sufficient to justify a depth result.
    /// @return false on fail else true.
    virtual bool getTextureDifferenceThreshold(uint32_t& textureDifferenceThresh) = 0;

    /// Not validated.
    /// Sets the threshold on how much the minimum correlation score must differ from the next best score.
    /// @param val how much worse the second peak must be.
    /// @return false on fail else true.
    virtual bool setSecondPeakThreshold(uint32_t val) = 0;
    /// Gets the threshold on how much the minimum correlation score must differ from the next best score.
    /// @return false on fail else true.
    virtual bool getSecondPeakThreshold(uint32_t& val) = 0;

    /// Not validated.
    /// Sends an i2c command to the imager(s) designated by DSWhichImagers. Register regAddress is given value regValue.
    /// @param DSWhichImagers one of DS_LEFT_IMAGER, DS_RIGHT_IMAGER DS_BOTH_IMAGERS
    /// @param regAddress  the i2c register address
    /// @param regValue  the value to set register regAddress to
    /// @param noCheck if true, do not check whether the write occurred correctly.
    /// @return false on fail else true.
    virtual bool writeImagerReg(DSWhichImager whichImager, uint16_t regAddress, uint16_t regValue, bool noCheck = false) = 0;
    /// Sends an i2c command to the imager designated by whichImagers. regValue is set to the contents of Register regAddress.
    /// @param DSWhichImagers one of DS_LEFT_IMAGER, DS_RIGHT_IMAGER DS_BOTH_IMAGERS
    /// @param regAddress  the i2c register address
    /// @param[out] regValue  where to put the value of register regAddress
    /// @return false on fail else true.
    virtual bool readImagerReg(DSWhichImager whichImager, uint16_t regAddress, uint16_t& regValue) = 0;

    /// Turn autogain/autoexposure on or off. Must be called after the camera has been initialized and capture has begun.
    virtual bool setAutoGainExposure(DSWhichImager whichImager, bool state) = 0;
    /// Returns true if autogain/autoexposure is enabled.
    virtual bool getAutoGainExposure(DSWhichImager whichImager, bool& state) = 0;

    /// Sets imager gain factor. For camera without internal auto exposure control.
    /// @param gain the gain factor
    /// @param which for which imager (or both) do we set the gain
    /// @return true for success, false on error.
    virtual bool setImagerGain(float gain, DSWhichImager whichImager) = 0;
    /// Gets imager gain factor. For camera without internal auto exposure control.
    /// @param[out] gain the gain factor.
    /// @param which for which imager do we get the gain.
    /// @return true for success, false on error.
    virtual bool getImagerGain(float& gain, DSWhichImager whichImager) = 0;
    /// Gets imager minimum gain factor. For camera without internal auto exposure control.
    /// @param[out] minGain the minimum gain factor.
    /// @param which for which imager do we get the minimum gain factor.
    /// @return true for success, false on error.
    virtual bool getImagerMinMaxGain(float& minGain, float& maxGain, DSWhichImager whichImager) = 0;

    /// Sets imager exposure time (in ms). For camera without internal auto exposure control.
    /// @param exposureTime the exposure time.
    /// @param which for which imager (or both) do we set the exposure time.
    /// @return true for success, false on error.
    virtual bool setImagerExposure(float exposure, DSWhichImager whichImager) = 0;
    /// Gets imager exposure time (in ms). For camera without internal auto exposure control.
    /// @param[out] exposure the exposure time.
    /// @param which for which imager do we get the exposure time.
    /// @return true for success, false on error.
    virtual bool getImagerExposure(float& exposure, DSWhichImager whichImager) = 0;
    /// Gets imager minimum exposure time (in ms). For camera without internal auto exposure control.
    /// @param[out] minExposure the minimum exposure time.
    /// @param which for which imager do we get the minimum exposure time.
    /// @return true for success, false on error.
    virtual bool getImagerMinMaxExposure(float& minExposure, float& maxExposure, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets temperature to describe the current temperture of the camera head.
    /// @param temperature floating point celsius temperature.
    /// @param whichSensor which temperature sensor.
    /// @return true for success, false on error.
    virtual bool readTemperature(float& temperature, DSWhichSensor whichSensor = DS_TEMPERATURE_SENSOR_1) = 0;
    virtual bool readMinMaxRecordedTemperature(float& minTemperature, float& maxTemperature, DSWhichSensor whichSensor = DS_TEMPERATURE_SENSOR_1) = 0;
    virtual bool resetMinMaxRecordedTemperture(DSWhichSensor whichSensor = DS_TEMPERATURE_SENSOR_1) = 0;

    /// Not validated.
    /// Sets Brightness
    virtual bool setBrightness(int val, DSWhichImager whichImager) = 0;
    virtual bool getBrightness(int& val, DSWhichImager whichImager) = 0;
    virtual bool getMinMaxBrightness(int& min, int& max, DSWhichImager whichImager) = 0;

    /// Not validated.
	/// Sets contrast, which is a value expressed as a gain factor multiplied by 100. Windows range is 0 - 10000, default is 100 (x1)
    virtual bool setContrast(int val, DSWhichImager whichImager) = 0;
    virtual bool getContrast(int& val, DSWhichImager whichImager) = 0;
    virtual bool getMinMaxContrast(int& min, int& max, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets saturation, which is a value expressed as a gain factor multiplied by 100. Windows range is 0 - 10000, default is 100 (x1)
    virtual bool setSaturation(int val, DSWhichImager whichImager) = 0;
    virtual bool getSaturation(int& val, DSWhichImager whichImager) = 0;
    virtual bool getMinMaxSaturation(int& min, int& max, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets hue, which is a value expressed as degrees multiplied by 100. Windows & UVC range is -18000 to 18000 (-180 to +180 degrees), default is 0
    virtual bool setHue(int val, DSWhichImager whichImager) = 0;
    virtual bool getHue(int& val, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets gamma, this is express as gamma multiplied by 100. Windows and UVC range is 1 to 500.
    virtual bool setGamma(int val, DSWhichImager whichImager) = 0;
    virtual bool getGamma(int& val, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets white balance. Color temperature, in degrees Kelvin. Windows has no defined range.  UVC range is 2800 (incandescent) to 6500 (daylight) but still needs to provide range
    virtual bool setWhiteBalance(int val, DSWhichImager whichImager) = 0;
    virtual bool getWhiteBalance(int& val, DSWhichImager whichImager) = 0;
    virtual bool getMinMaxWhiteBalance(int& min, int& max, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets sharpness. Arbitrary units. UVC has no specified range (min sharpness means no sharpening), Windows required range must be 0 through 100. The default value must be 50.
    virtual bool setSharpness(int val, DSWhichImager whichImager) = 0;
    virtual bool getSharpness(int& val, DSWhichImager whichImager) = 0;
    virtual bool getMinMaxSharpness(int& min, int& max, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets back light compensation. A value of false indicates that the back-light compensation is disabled. The default value of true indicates that the back-light compensation is enabled.
    virtual bool setBacklightCompensation(bool val, DSWhichImager whichImager) = 0;
    virtual bool getBacklightCompensation(bool& val, DSWhichImager whichImager) = 0;

    /// Not validated.
    /// Sets third power line frequency.
    virtual bool setThirdPowerLineFrequency(DSPowerLineFreqOption plf) = 0;
    virtual bool getThirdPowerLineFrequency(DSPowerLineFreqOption& plf) = 0;

protected:
    // Creation (and deletion) of an object of this
    // type is supported through the DSFactory functions.
    DSHardware() {};
    DSHardware(const DSHardware& other) DS_DELETED_FUNCTION;
    DSHardware(DSHardware&& other) DS_DELETED_FUNCTION;
    DSHardware& operator=(const DSHardware& other) DS_DELETED_FUNCTION;
    DSHardware& operator=(DSHardware&& other) DS_DELETED_FUNCTION;
    virtual ~DSHardware() {};
};
