#pragma once

/**
* Converts meters to z units.
* Converts meters to units output in the Z image based on current settings. Allows your code
* to be independent of the choice of z units.
*
* @param mts meters
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to mts meters
* @see DSAPI methods setZUnits, getZUnits
*/
inline int DSConvertMToZUnits(double mts, int zUnits)
{
    return static_cast<int>(mts * 1000000 / zUnits);
}

/**
* Converts centimeters to z units.
* Converts centimeters to units output in the Z image based on current settings. Allows your
* code to be independent of the choice of z units.
* 
* @param cms centimeters
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to cms centimeters
* @see DSAPI setZUnits, getZUnits
*/
inline int DSConvertCMToZUnits(double cms, int zUnits)
{
    return static_cast<int>(cms * 10000 / zUnits);
}

/**
* Converts millimeters to z units.
* Converts millimeters to units output in the Z image based on current settings. Allows your
* code to be independent of the choice of z units.
*
* @param mms millimeters
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to mms millimeters
* @see DSAPI setZUnits, getZUnits
*/
inline int DSConvertMMToZUnits(double mms, int zUnits)
{
    return static_cast<int>(mms * 1000 / zUnits);
}

/**
* Converts yards to z units.
* Converts yards to units output in the Z image based on current settings. Allows your
* code to be independent of the choice of z units.
* 
* @param yds yards
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to yds yards
* @see DSAPI setZUnits, getZUnits
*/
inline int DSConvertYdToZUnits(double yds, int zUnits)
{
    return static_cast<int>(yds * 914400 / zUnits);
}

/**
* Converts feet to z units.
* Converts inches to units output in the Z image based on current settings. Allows your
* code to be independent of the choice of z units.
*
* @param ft feet
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to ft feet
* @see DSAPI setZUnits, getZUnits
*/
inline int DSConvertFtToZUnits(double ft, int zUnits)
{
    return static_cast<int>(ft * 304800 / zUnits);
}

/**
* Converts inches to z units.
* Converts inches to units output in the Z image based on current settings. Allows your
* code to be independent of the choice of z units.
*
* @param ins inches
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return number of z units corresponding to ins inches
* @see DSAPI setZUnits, getZUnits
*/
inline int DSConvertInToZUnits(double ins, int zUnits)
{
    return static_cast<int>(ins * 25400 / zUnits);
}

/**
* Converts z units to meters.
* Converts units output in the Z image to meters. Allows your code to be independent of 
* the choice of z units.
* 
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return meters corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToM(int z, int zUnits)
{
    return z * zUnits / 1000000.0;
}

/**
* Converts z units to centimeters.
* Converts units output in the Z image to centimeters. Allows your	code to be independent 
* of the choice of z units.
* 
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return centimeters corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToCM(int z, int zUnits)
{
    return z * zUnits / 10000.0;
}

/**
* Converts z units to millimeters.
* Converts units output in the Z image to millimeters. Allows your	code to be independent 
* of the choice of z units.
* 
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return millimeters corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToMM(int z, int zUnits)
{
    return z * zUnits / 1000.0;
}

/**
* Converts z units to yards.
* Converts units output in the Z image to yards. Allows your code to be independent of the choice of z units.
*
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return yards corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToYd(int z, int zUnits)
{
    return z * zUnits / 914400.0;
}

/**
* Converts z units to feet.
* Converts units output in the Z image to feet. Allows your code to be independent of the choice of z units.
* 
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return feet corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToFt(int z, int zUnits)
{
    return z * zUnits / 304800.0;
}

/**
* Converts z units to inches.
* Converts units output in the Z image to inches. Allows your code to be independent of the choice of z units.
* 
* @param z value
* @param zUnits, units of the Z image in micrometers, get this from DSAPI.getZUnits()
* @return inches corresponding to z units
* @see DSAPI setZUnits, getZUnits
*/
inline double DSConvertZUnitsToIn(int z, int zUnits)
{
    return z * zUnits / 25400.0;
}
