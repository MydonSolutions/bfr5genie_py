import os, sys, argparse, glob
import numpy
from astropy.coordinates import SkyCoord

import bfr5genie

def _base_arguments_parser():
    parser = argparse.ArgumentParser(
        description="A script that generates a BFR5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--telescope-info-toml-filepath",
        type=str,
        required=True,
        help="The path to telescope information.",
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default=None,
        help="The path to which the output will be written (instead of alongside the raw_filepath).",
    )
    parser.add_argument(
        "-m",
        "--mitigate-antenna",
        default=[],
        nargs="+",
        metavar=("antname"),
        help="The names of antenna to mitigate (associated calibration coefficients are set to zero)."
    )
    parser.add_argument(
        "-p",
        "--phase-center",
        default=None,
        metavar=("ra_hour,dec_deg"),
        help="The coordinates of the data's phase-center."
    )
    parser.add_argument(
        "--ata-raw",
        action="store_true",
        help="Drop the last character of the RAW file's antenna names (the ATA antenna names have an L.O. character suffix)."
    )
    parser.add_argument(
        "raw_filepaths",
        type=str,
        nargs="+",
        help="The path to the GUPPI RAW file stem or of all files.",
    )
    return parser

def _parse_base_arguments(args):
    
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        bfr5genie.logger.info(f"Given RAW filepath does not exist, assuming it is the stem.")
        args.raw_filepaths = glob.glob(f"{args.raw_filepaths[0]}*.raw")
        bfr5genie.logger.info(f"Found {args.raw_filepaths}.")

    raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center = bfr5genie.get_raw_metadata(args.raw_filepaths, raw_antname_callback= None if not args.ata_raw else lambda x: x[:-1])

    telinfo = bfr5genie.get_telescope_metadata(args.telescope_info_toml_filepath)

    input_dir, input_filename = os.path.split(args.raw_filepaths[0])
    if args.output_filepath is None:
        output_filepath = os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.bfr5")
    else:
        output_filepath = args.output_filepath
    bfr5genie.logger.info(f"Output filepath: {output_filepath}")

    if args.phase_center is not None:
        (phase_center_ra, phase_center_dec) = args.phase_center.split(',')
        phase_center = bfr5genie.SkyCoord(
            float(phase_center_ra) * numpy.pi / 12.0 ,
            float(phase_center_dec) * numpy.pi / 180.0 ,
            unit='rad'
        )
    
    
    npol, nants = raw_header["NPOL"], len(antenna_names)
    calcoeff_bandpass = numpy.ones(
        (len(frequencies_hz), npol, nants)
    )*(1+0j)

    calcoeff_gain = numpy.ones(
        (npol, nants)
    )*(1+0j)
    for ant_name in args.mitigate_antenna:
        if ant_name not in antenna_names:
            raise RuntimeError(f"Cannot mitigate antenna {ant_name} as it is not recorded in the file. Antenna names: {antenna_names}")
        index = antenna_names.index(ant_name)
        calcoeff_gain[:, index] = 0

    return raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain

def _add_arguments_targetselector(parser):
    parser.add_argument(
        "--redis-hostname",
        type=str,
        default="redishost",
        help="The hostname of the Redis server.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="The port of the Redis server.",
    )
    parser.add_argument(
        "--targets-redis-key-prefix",
        type=str,
        default="targets:VLA-COSMIC:vlass_array",
        help="The prefix of the redis key holding targets in JSON."
    )
    parser.add_argument(
        "--targets-redis-key-timestamp",
        type=str,
        default=None,
        help="The identifying timestamp of the redis key holding targets in JSON. If not supplied, it is ascertained from the starting time of the raw file."
    )
    parser.add_argument(
        "--take-targets",
        type=int,
        default=5,
        help="The number of targets to form beams on from the redis key."
    )
    parser.add_argument(
        "--take-targets-after",
        type=int,
        default=1,
        help="The number of targets to skip. Typically 1 to skip the first target which is phase-center."
    )


def _add_arguments_raster(parser):
    parser.add_argument(
        "--raster-ra",
        required=True,
        metavar=("ra_offset_start", "ra_offset_stop", "ra_offset_step"),
        nargs=3,
        type=float,
        help="The phase-center relative right-ascension range (in hours) for a raster set of beams."
    )
    parser.add_argument(
        "--raster-dec",
        required=True,
        metavar=("dec_offset_start", "dec_offset_stop", "dec_offset_step"),
        nargs=3,
        type=float,
        help="The phase-center relative declination range (in degrees) for a raster set of beams."
    )


def _generate_bfr5_for_raw(
    raw_header,
    antenna_names,
    frequencies_hz,
    times_unix,
    phase_center,
    primary_center,
    telinfo,
    output_filepath,
    calcoeff_bandpass,
    calcoeff_gain,
    beam_strs
):
    
    beams = {}
    for i, beam_str in enumerate(beam_strs):
        coords = beam_str.split(',')
        if len(coords) == 3:
            beam_name = coords[-1]
        else:
            beam_name = f"BEAM_{i}"
        beams[beam_name] = SkyCoord(
            float(coords[0]) * numpy.pi / 12.0,
            float(coords[1]) * numpy.pi / 180.0,
            unit='rad'
        )

    nbeams = len(beams)
    if nbeams == 0:
        bfr5genie.logger.warning(f"No beam coordinates provided, forming a beam on phase-center.")
        beams["PHASE_CENTER"] = phase_center
        nbeams = 1

    beam_coordinates_string = "\n\t".join(f"{k}: {v}".replace("\n", "") for k, v in beams.items())
    bfr5genie.logger.info(f"Beam coordinates:\n\t{beam_coordinates_string}")

    telescope_antenna_names = set([antenna["name"] for antenna in telinfo["antennas"] if antenna["name"] in antenna_names])
    assert len(telescope_antenna_names) == len(antenna_names), f"Telescope information does not cover RAW listed antenna: {set(antenna_names).difference(telescope_antenna_names)}"

    bfr5genie.write(
        output_filepath,
        raw_header.get("OBSID", "Unknown"),
        telinfo["telescope_name"],
        "Unknown Instrument",
        beams,
        phase_center,
        (telinfo["longitude"], telinfo["latitude"], telinfo["altitude"]),
        [antenna for antenna in telinfo["antennas"] if antenna["name"] in antenna_names],
        times_unix,
        frequencies_hz,
        calcoeff_bandpass,
        calcoeff_gain,
        dut1 = raw_header.get("DUT1", 0.0),
        primary_center = primary_center,
        reference_antenna = raw_header.get("REFANT", None)
    )
    bfr5genie.logger.info(output_filepath)


def generate_targets_for_raw(arg_values=None):
    parser = _base_arguments_parser()
    _add_arguments_targetselector(parser)
    args = parser.parse_args(arg_values if arg_values is not None else sys.argv[1:])
    
    raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

    beam_strs = []
    redis_obj = redis.Redis(host=args.redis_hostname, port=args.redis_port)
    if args.targets_redis_key_timestamp is None:
        file_start_packet_index = raw_header["SYNCTIME"] + raw_header["PKTIDX"]
        bfr5genie.logger.info(f"Targets key timestamp is taken to be the starting packet-index of the file: {file_start_packet_index}")
        args.targets_redis_key_timestamp = file_start_packet_index

    targets_redis_key = f"{args.targets_redis_key_prefix}:{args.targets_redis_key_timestamp}"
    bfr5genie.logger.info(f"Accessing targets at {targets_redis_key}.")
    targets = redis_obj.get(targets_redis_key)
    if targets is None:
        raise ValueError(f"No targets to retrieve at: {targets_redis_key}.")

    targets = json.loads(targets)

    for target in targets[args.take_targets_after : args.take_targets_after+args.take_targets]:
        beam_strs.append(
            f"{target['ra']*24/360},{target['dec']},{target['source_id']}"
        )

    if len(beam_strs) < args.take_targets:
        bfr5genie.logger.warning(f"Could only take {len(beam_strs)} targets.")

    _generate_bfr5_for_raw(
        raw_header,
        antenna_names,
        frequencies_hz,
        times_unix,
        phase_center,
        primary_center,
        telinfo,
        output_filepath,
        calcoeff_bandpass,
        calcoeff_gain,
        beam_strs
    )

def generate_raster_for_raw(arg_values=None):
    parser = _base_arguments_parser()
    _add_arguments_raster(parser)
    args = parser.parse_args(arg_values if arg_values is not None else sys.argv[1:])
    
    raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

    beam_strs = []
    raster_args = [args.raster_ra, args.raster_dec]
    assert all(raster_args), f"Must supply both raster arguments for raster beams to be generated"

    for ra_index, ra in enumerate(numpy.arange(*args.raster_ra)):
        for dec_index, dec in enumerate(numpy.arange(*args.raster_dec)):
            beam_strs.append(f"{phase_center.ra.deg*12.0/180.0 + ra},{phase_center.dec.deg + dec},raster_{ra_index}_{dec_index}")

    _generate_bfr5_for_raw(
        raw_header,
        antenna_names,
        frequencies_hz,
        times_unix,
        phase_center,
        primary_center,
        telinfo,
        output_filepath,
        calcoeff_bandpass,
        calcoeff_gain,
        beam_strs
    )

def generate_for_raw(arg_values=None):
    parser = _base_arguments_parser()
    parser.add_argument(
        "-b",
        "--beam",
        default=None,
        action="append",
        metavar=("ra_hour,dec_deg[,name]"),
        help="The coordinates of a beam (optionally the name too)."
    )
    args = parser.parse_args(arg_values if arg_values is not None else sys.argv[1:])

    raw_header, antenna_names, frequencies_hz, times_unix, phase_center, primary_center, telinfo, output_filepath, calcoeff_bandpass, calcoeff_gain = _parse_base_arguments(args)

    beam_strs = []
    if args.beam is None:
        # scrape from RAW file RA_OFF%01d,DEC_OFF%01d
        key_enum = 0
        while True:
            ra_key = f"RA_OFF{key_enum}"
            dec_key = f"DEC_OFF{key_enum}"
            if not (ra_key in raw_header and dec_key in raw_header):
                break

            beam_strs.append(f"{raw_header[ra_key]},{raw_header[dec_key]},BEAM_{key_enum}")

            key_enum += 1
            if key_enum == 10:
                break

        bfr5genie.logger.info(f"Collected {key_enum} beam coordinates from the RAW header, in lieu of CLI provided beam coordinates.")
    elif len(args.beam) > 0:
        bfr5genie.logger.info(args.beam)
        beam_strs = list(b for b in args.beam)

    _generate_bfr5_for_raw(
        raw_header,
        antenna_names,
        frequencies_hz,
        times_unix,
        phase_center,
        primary_center,
        telinfo,
        output_filepath,
        calcoeff_bandpass,
        calcoeff_gain,
        beam_strs
    )