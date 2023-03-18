import logging, json
from datetime import datetime

import numpy
import pyproj

import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time

import erfa

import h5py
import tomli as tomllib # `tomllib` as of Python 3.11 (PEP 680)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)

def _degrees_process(value):
    logger.debug(f"_degrees_process input: {value}")
    if isinstance(value, str):
        if value.count(':') == 2:
            value = value.split(':')
            value_f = float(value[0])
            if value_f < 0:
                value_f -= (float(value[1]) + float(value[2])/60)/60
            else:
                value_f += (float(value[1]) + float(value[2])/60)/60
            logger.debug(f"_degrees_process output: {value_f}")
            return value_f
    logger.debug(f"_degrees_process output: {float(value)}")
    return float(value)

def transform_antenna_positions_ecef_to_xyz(longitude_deg, latitude_deg, altitude, antenna_positions):
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
    )
    telescopeCenterXyz = transformer.transform(
        longitude_deg,
        latitude_deg,
        altitude,
    )
    logger.debug(f"LLA ({(longitude_deg,latitude_deg,altitude)}) converted to ECEF-XYZ ({telescopeCenterXyz})")
    for i in range(antenna_positions.shape[0]):
        antenna_positions[i, :] -= telescopeCenterXyz


def transform_antenna_positions_xyz_to_enu(longitude_rad, latitude_rad, altitude, antenna_positions):
    sin_long = numpy.sin(longitude_rad)
    cos_long = numpy.cos(longitude_rad)
    sin_lat = numpy.sin(latitude_rad)
    cos_lat = numpy.cos(latitude_rad)

    enus = numpy.zeros(antenna_positions.shape, dtype=numpy.float64)

    for ant in range(antenna_positions.shape[0]):
        # RotZ(longitude) anti-clockwise
        x = cos_long*antenna_positions[ant, 0] - (-sin_long)*antenna_positions[ant, 1]
        y = (-sin_long)*antenna_positions[ant, 0] + cos_long*antenna_positions[ant, 1]
        z = antenna_positions[ant, 2]

        # RotY(latitude) clockwise
        x_ = x
        x = cos_lat*x_ + sin_lat*z
        z = -sin_lat*x_ + cos_lat*z

        # Permute (UEN) to (ENU)
        enus[ant, 0] = y
        enus[ant, 1] = z
        enus[ant, 2] = x

    return enus


def _compute_uvw_from_enu(ts, source, ant_enu_coordinates, lla, dut1=0.0, astrom=None):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        ts: array of Times to compute the coordinates
        source: source SkyCoord
        ant_enu_coordinates: numpy.ndarray
            Antenna XYZ coordinates, relative to reference position. This is indexed as (antenna_number, xyz)
        lla: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.
        astrom: erfa.astrom
            erfa.apco13 generated astrom value to reuse.

    Returns:
        The UVW coordinates in metres of each antenna. This
        is indexed as (antenna_number, uvw)
    """

    if astrom is not None:
        ri, di = erfa.atciq(
            source.ra.rad, source.dec.rad,
            0, 0, 0, 0,
            astrom
        )
        aob, zob, ha_rad, dec_rad, rob = erfa.atioq(
            ri, di,
            astrom
        )
    else:
        aob, zob, ha_rad, dec_rad, rob, eo = erfa.atco13(
            source.ra.rad, source.dec.rad,
            0, 0, 0, 0,
            time_jd, 0,
            dut1,
            *lla,
            0, 0,
            0, 0, 0, 0
        )
        
    sin_hangle = numpy.sin(ha_rad)
    cos_hangle = numpy.cos(ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)
    sin_latitude = numpy.sin(lla[1])
    cos_latitude = numpy.cos(lla[1])

    uvws = numpy.zeros(ant_enu_coordinates.shape, dtype=numpy.float64)

    for ant in range(ant_enu_coordinates.shape[0]):
        # RotX(latitude) anti-clockwise
        x = ant_enu_coordinates[ant, 0]
        y = cos_latitude*ant_enu_coordinates[ant, 1] - (-sin_latitude)*ant_enu_coordinates[ant, 2]
        z = (-sin_latitude)*ant_enu_coordinates[ant, 1] + cos_latitude*ant_enu_coordinates[ant, 2]

        # RotY(hour_angle) clockwise
        x_ = x
        x = cos_hangle*x_ + sin_hangle*z
        z = -sin_hangle*x_ + cos_hangle*z

        # RotX(declination) clockwise
        uvws[ant, 0] = x
        uvws[ant, 1] = cos_declination*y - sin_declination*z
        uvws[ant, 2] = sin_declination*y + cos_declination*z

    return uvws

def _compute_uvw(ts, source, ant_coordinates, lla, dut1=0.0, astrom=None):
    """Computes UVW antenna coordinates with respect to reference

    Args:
        ts: array of Times to compute the coordinates
        source: source SkyCoord
        ant_coordinates: numpy.ndarray
            Antenna XYZ coordinates, relative to reference position. This is indexed as (antenna_number, xyz)
        lla: tuple Reference Coordinates (radians)
            Longitude, Latitude, Altitude. The antenna_coordinates must have
            this component in them.
        astrom: erfa.astrom
            erfa.apco13 generated astrom value to reuse.

    Returns:
        The UVW coordinates in metres of each antenna. This
        is indexed as (antenna_number, uvw)
    """

    if astrom is not None:
        ri, di = erfa.atciq(
            source.ra.rad, source.dec.rad,
            0, 0, 0, 0,
            astrom
        )
        aob, zob, ha_rad, dec_rad, rob = erfa.atioq(
            ri, di,
            astrom
        )
    else:
        aob, zob, ha_rad, dec_rad, rob, eo = erfa.atco13(
            source.ra.rad, source.dec.rad,
            0, 0, 0, 0,
            ts.jd, 0,
            dut1,
            *lla,
            0, 0,
            0, 0, 0, 0
        )
        
    sin_long_minus_hangle = numpy.sin(lla[0]-ha_rad)
    cos_long_minus_hangle = numpy.cos(lla[0]-ha_rad)
    sin_declination = numpy.sin(dec_rad)
    cos_declination = numpy.cos(dec_rad)

    uvws = numpy.zeros(ant_coordinates.shape, dtype=numpy.float64)

    for ant in range(ant_coordinates.shape[0]):
        # RotZ(long-ha) anti-clockwise
        x = cos_long_minus_hangle*ant_coordinates[ant, 0] - (-sin_long_minus_hangle)*ant_coordinates[ant, 1]
        y = (-sin_long_minus_hangle)*ant_coordinates[ant, 0] + cos_long_minus_hangle*ant_coordinates[ant, 1]
        z = ant_coordinates[ant, 2]

        # RotY(declination) clockwise
        x_ = x
        x = cos_declination*x_ + sin_declination*z
        z = -sin_declination*x_ + cos_declination*z

        # Permute (WUV) to (UVW)
        uvws[ant, 0] = y
        uvws[ant, 1] = z
        uvws[ant, 2] = x

    return uvws


def phasor_delays(
    antennaPositions: numpy.ndarray, # [Antenna, XYZ] relative to whatever reference position
    boresightCoordinate: SkyCoord, # ra-dec
    beamCoordinates: 'list[SkyCoord]', #  ra-dec
    times: numpy.ndarray, # [unix]
    lla: tuple, # Longitude, Latitude, Altitude (radians)
    referenceAntennaIndex: int = 0,
    dut1: float = 0.0
):
    """
    Return
    ------
        delays_ns (T, A, B)
    """

    delays_ns = numpy.zeros(
        (
            times.shape[0],
            beamCoordinates.shape[0],
            antennaPositions.shape[0],
        ),
        dtype=numpy.float64
    )

    for t, tval in enumerate(times):
        ts = Time(tval, format='unix')

        # get valid eraASTROM instance
        astrom, eo = erfa.apco13(
            ts.jd, 0,
            dut1,
            *lla,
            0, 0,
            0, 0, 0, 0
        )

        boresightUvw = _compute_uvw(
            ts,
            boresightCoordinate,
            antennaPositions,
            lla,
            astrom = astrom
        )
        boresightUvw -= boresightUvw[referenceAntennaIndex:referenceAntennaIndex+1, :]
        for b, beam_coord in enumerate(beamCoordinates):
            # These UVWs are centered at the reference antenna,
            # i.e. UVW_irefant = [0, 0, 0]
            beamUvw = _compute_uvw( # [Antenna, UVW]
                ts,
                beam_coord,
                antennaPositions,
                lla,
                astrom = astrom
            )
            beamUvw -= beamUvw[referenceAntennaIndex:referenceAntennaIndex+1, :]

            delays_ns[t, b, :] = (beamUvw[:,2] - boresightUvw[:,2]) * (1e9 / const.c.value)

    return delays_ns


def phasors_from_delays(
    delays_ns: numpy.ndarray, # [Time, Beam, Antenna]
    frequencies: numpy.ndarray, # [channel-frequencies] Hz
    calibrationCoefficients: numpy.ndarray, # [Frequency-channel, Polarization, Antenna]
):
    """
    Return
    ------
        phasors (B, A, F, T, P)
    """

    assert frequencies.shape[0] % calibrationCoefficients.shape[0] == 0, f"Calibration Coefficients' Frequency axis is not a factor of frequencies: {calibrationCoefficients.shape[0]} vs {frequencies.shape[0]}."

    phasorDims = (
        delays_ns.shape[1],
        delays_ns.shape[2],
        frequencies.shape[0],
        delays_ns.shape[0],
        calibrationCoefficients.shape[1]
    )
    calibrationCoeffFreqRatio = frequencies.shape[0] // calibrationCoefficients.shape[0]
    calibrationCoefficients = numpy.repeat(calibrationCoefficients, calibrationCoeffFreqRatio, axis=0) # repeat frequencies

    phasors = numpy.zeros(phasorDims, dtype=numpy.complex128)

    for t in range(delays_ns.shape[0]):
        for b in range(delays_ns.shape[1]):
            for a, delay_ns in enumerate(delays_ns[t, b, :]):
                phasor = numpy.exp(-1.0j*2.0*numpy.pi*delay_ns*1e-9*frequencies)
                phasors[b, a, :, t, :] = numpy.reshape(numpy.repeat(phasor, 2), (len(phasor), 2)) * calibrationCoefficients[:, :, a]
    return phasors


def get_telescope_metadata(telescope_info_toml_filepath):
    """
    Returns a standardised formation of the TOML contents:
    {
        "telescope_name": str,
        "longitude": float, # radians
        "latitude": float, # radians
        "altitude": float,
        "antenna_position_frame": "xyz", # meters relative to lla
        "antennas": [
            {
                "name": str,
                "position": [X, Y, Z],
                "number": int,
                "diameter": float,
            }
        ]
    }
    """

    with open(telescope_info_toml_filepath, mode="rb") as f:
        telescope_info = tomllib.load(f)

    longitude = _degrees_process(telescope_info["longitude"])
    latitude = _degrees_process(telescope_info["latitude"])
    altitude = telescope_info["altitude"]
    antenna_positions = numpy.array([antenna["position"] for antenna in telescope_info["antennas"]])

    if "ecef" == telescope_info["antenna_position_frame"].lower():
        logger.info("Transforming antenna positions from XYZ to ECEF.")
        transform_antenna_positions_ecef_to_xyz(
            longitude,
            latitude,
            altitude,
            antenna_positions,
        )
    else:
        # TODO handle enu
        assert telescope_info["antenna_position_frame"].lower() == "xyz"
        logger.info("Taking verbatim XYZ antenna positions.")

    return {
        "telescope_name": telescope_info["telescope_name"],
        "longitude": longitude*numpy.pi/180.0,
        "latitude": latitude*numpy.pi/180.0,
        "altitude": altitude,
        "antenna_position_frame": "xyz",
        "antennas": [
            {
                "name": ant_info["name"],
                "position": antenna_positions[ant_enum],
                "number": ant_info.get("number", ant_enum),
                "diameter": ant_info.get("diameter", telescope_info["antenna_diameter"]),
            }
            for ant_enum, ant_info in enumerate(telescope_info["antennas"])
        ]
    }


def get_raw_metadata(raw_filepaths, raw_antname_callback=None):
    raw_blocks = 0
    raw_header = {}
    for raw_file_enum, raw_filepath in enumerate(raw_filepaths):
        with open(raw_filepath, mode="rb") as f:
            header_entry = f.read(80).decode()
            while header_entry:
                if header_entry == "END" + " "*77:
                    break

                if raw_file_enum != 0:
                    header_entry = f.read(80).decode()
                    continue

                key = header_entry[0:8].strip()
                value = header_entry[9:].strip()
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except:
                    # must be a str value, drop enclosing single-quotes
                    assert value[0] == value[-1] == "'"
                    value = value[1:-1].strip()

                raw_header[key] = value
                header_entry = f.read(80).decode()

            logger.debug(f"{raw_filepath} First header: {raw_header} (ends @ position {f.tell()})")

            # count number of blocks in file, assume BLOCSIZE is consistent
            data_seek_size = raw_header["BLOCSIZE"]
            if raw_header.get("DIRECTIO", 0) == 1:
                data_seek_size = int((data_seek_size + 511) / 512) * 512
                logger.debug(f"BLOCSIZE rounded {raw_header['BLOCSIZE']} up to {data_seek_size}")

            while True:
                if raw_header.get("DIRECTIO", 0) == 1:
                    origin = f.tell()
                    f.seek(int((f.tell() + 511) / 512) * 512)
                    logger.debug(f"Seeked past padding: {origin} -> {f.tell()}")

                f.seek(data_seek_size + f.tell())
                block_header_start = f.tell()
                raw_blocks += 1
                try:
                    header_entry = f.read(80).decode()
                    if len(header_entry) < 80:
                        break
                    while header_entry != "END" + " "*77:
                        header_entry = f.read(80).decode()
                except UnicodeDecodeError as err:
                    pos = f.tell()
                    f.seek(pos - 321)
                    preceeding_bytes = f.read(240)
                    next_bytes = f.read(240)

                    logger.error(f"UnicodeDecodeError in {raw_filepath} at position: {pos}")
                    logger.error(f"Preceeding bytes: {preceeding_bytes}")
                    logger.error(f"Proceeding bytes: {next_bytes}")
                    logger.error(f"Block #{raw_blocks} starting at {block_header_start}")

                    raise RuntimeError(f"Failed to iterate through RAW files.")

    # General RAW metadata
    nants = raw_header.get("NANTS", 1)
    npol = raw_header["NPOL"]
    nchan = raw_header["OBSNCHAN"] // nants
    schan = raw_header.get("SCHAN", 0)
    ntimes = (raw_header["BLOCSIZE"] * 8) // (raw_header["OBSNCHAN"] * raw_header["NPOL"] * 2 * raw_header["NBITS"])
    antenna_names = []
    for i in range(100):
        key = f"ANTNMS{i:02d}"
        if key in raw_header:
            if raw_antname_callback is not None:
                antenna_names += map(raw_antname_callback, raw_header[key].split(","))
            else:
                antenna_names += raw_header[key].split(",")

    antenna_names[0:nants]

    start_time_unix = raw_header["SYNCTIME"] + raw_header["PKTIDX"] * raw_header.get("TBIN", 1/raw_header["CHAN_BW"]) * ntimes/raw_header.get("PIPERBLK", ntimes)
    block_time_span_s = raw_header.get("PIPERBLK", ntimes) * raw_header.get("TBIN", 1/raw_header["CHAN_BW"]) * ntimes/raw_header.get("PIPERBLK", ntimes)

    phase_center = SkyCoord(
        float(raw_header.get("RA_PHAS", raw_header["RA_STR"])) * numpy.pi / 12.0 ,
        float(raw_header.get("DEC_PHAS", raw_header["DEC_STR"])) * numpy.pi / 180.0 ,
        unit='rad'
    )

    primary_center = SkyCoord(
        float(raw_header.get("RA_PRIM", raw_header["RA_STR"])) * numpy.pi / 12.0 ,
        float(raw_header.get("DEC_PRIM", raw_header["DEC_STR"])) * numpy.pi / 180.0 ,
        unit='rad'
    )

    # find the observation channel0 frequency, which is the start of the frequency range
    frequency_channel_0_hz = raw_header["OBSFREQ"] - (nchan/2 + schan)*raw_header["CHAN_BW"]
    frequencies_hz = (frequency_channel_0_hz + numpy.arange(schan+nchan)*raw_header["CHAN_BW"])*1e6
    mid_chan = schan + (nchan//2)
    assert frequencies_hz[mid_chan] == raw_header["OBSFREQ"]*1e6, f"frequencies_hz[{mid_chan}] = {mid_chan} != {raw_header['OBSFREQ']*1e6} (OBSFREQ)"

    times_unix = (start_time_unix + 0.5 * block_time_span_s) + numpy.arange(raw_blocks)*block_time_span_s

    return raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center


def write(
  output_filepath,
  obs_id: str,
  telescope_name: str,
  instrument_name: str,
  beam_src_coord_map: "dict(str, SkyCoord)",
  phase_center: SkyCoord,
  reference_lla: tuple, # (longitude:radians, latitude:radians, altitude)
  antennas: "list(dict)", # {position, name, diameter, number}
  times_unix: numpy.ndarray,
  frequencies_hz: numpy.ndarray, # (nchan)
  calcoeff_bandpass: numpy.ndarray, # (nchan, npol, nants)
  calcoeff_gain: numpy.ndarray, # (npol, nants)
  dut1: float = 0.0,
  primary_center: SkyCoord = None,
  reference_antenna: str = None
):
    nants = len(antennas)
    npol = calcoeff_gain.shape[0]
    nchan = len(frequencies_hz)
    nbeams = len(beam_src_coord_map)
    ntimes = len(times_unix)

    calcoeff = calcoeff_bandpass * numpy.reshape(numpy.repeat(calcoeff_gain, nchan, axis=0), (nchan, npol, nants))

    antenna_names = [ant["name"] for ant in antennas]
    if reference_antenna is None:
        reference_antenna = antenna_names[0]

    antenna_positions = numpy.array([ant["position"] for ant in antennas])
    delay_ns = phasor_delays(
        antenna_positions,
        phase_center,
        numpy.array(list(beam_src_coord_map.values())),
        times_unix,
        reference_lla,
        referenceAntennaIndex = antenna_names.index(reference_antenna)
    )

    with h5py.File(output_filepath, "w") as f:
        dimInfo = f.create_group("diminfo")
        dimInfo.create_dataset("nants", data=nants)
        dimInfo.create_dataset("npol", data=npol)
        dimInfo.create_dataset("nchan", data=nchan)
        dimInfo.create_dataset("nbeams", data=nbeams)
        dimInfo.create_dataset("ntimes", data=ntimes)

        beamInfo = f.create_group("beaminfo")
        beamInfo.create_dataset("ras", data=numpy.array([beam.ra.rad for beam in beam_src_coord_map.values()]), dtype='d') # radians
        beamInfo.create_dataset("decs", data=numpy.array([beam.dec.rad for beam in beam_src_coord_map.values()]), dtype='d') # radians
        source_names = [beam.encode() for beam in beam_src_coord_map.keys()]
        longest_source_name = max(len(name) for name in source_names)
        beamInfo.create_dataset("src_names", data=numpy.array(source_names, dtype=f"S{longest_source_name}"), dtype=h5py.special_dtype(vlen=str))

        calInfo = f.create_group("calinfo")
        calInfo.create_dataset("refant", data=reference_antenna.encode())
        calInfo.create_dataset("cal_K", data=numpy.ones((npol, nants)), dtype='d')
        calInfo.create_dataset("cal_B", data=calcoeff_bandpass, dtype='D')
        calInfo.create_dataset("cal_G", data=calcoeff_gain, dtype='D')
        calInfo.create_dataset("cal_all", data=calcoeff, dtype='D')

        delayInfo = f.create_group("delayinfo")
        delayInfo.create_dataset("delays", data=delay_ns, dtype='d')
        delayInfo.create_dataset("rates", data=numpy.zeros((ntimes, nbeams, nants)), dtype='d')
        delayInfo.create_dataset("time_array", data=times_unix, dtype='d')
        delayInfo.create_dataset("jds", data=(times_unix/86400) + 2440587.5, dtype='d')
        delayInfo.create_dataset("dut1", data=dut1, dtype='d')

        obsInfo = f.create_group("obsinfo")
        obsInfo.create_dataset("obsid", data=obs_id.encode())
        obsInfo.create_dataset("freq_array", data=frequencies_hz*1e-9, dtype='d') # GHz
        obsInfo.create_dataset("phase_center_ra", data=phase_center.ra.rad, dtype='d') # radians
        obsInfo.create_dataset("phase_center_dec", data=phase_center.dec.rad, dtype='d') # radians
        if primary_center is not None:
            obsInfo.create_dataset("primary_center_ra", data=primary_center.ra.rad, dtype='d') # radians
            obsInfo.create_dataset("primary_center_dec", data=primary_center.dec.rad, dtype='d') # radians
        obsInfo.create_dataset("instrument_name", data=instrument_name.encode())

        telInfo = f.create_group("telinfo")
        telInfo.create_dataset("antenna_positions", data=antenna_positions, dtype='d')
        telInfo.create_dataset("antenna_position_frame", data="xyz".encode())
        longest_antenna_name = max(*[len(name) for name in antenna_names])
        telInfo.create_dataset("antenna_names", data=numpy.array(antenna_names, dtype=f"S{longest_antenna_name}"), dtype=h5py.special_dtype(vlen=str))
        telInfo.create_dataset("antenna_numbers", data=numpy.array([ant["number"] for ant in antennas]), dtype='i')
        telInfo.create_dataset("antenna_diameters", data=numpy.array([ant["diameter"] for ant in antennas]), dtype='d')
        telInfo.create_dataset("longitude", data=reference_lla[0]*180/numpy.pi, dtype='d') # degrees
        telInfo.create_dataset("latitude", data=reference_lla[1]*180/numpy.pi, dtype='d') # degrees
        telInfo.create_dataset("altitude", data=reference_lla[2], dtype='d')
        telInfo.create_dataset("telescope_name", data=telescope_name.encode())
