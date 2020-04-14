import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from struct import unpack_from
from typing import Tuple

import numpy as np


class AbstractSpectralIO(ABC):
    file_path: str = None
    metadata: dict = None

    @abstractmethod
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        pass

    @abstractmethod
    def get_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int]) -> np.ndarray:
        pass

    @abstractmethod
    def set_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int],
                         data: np.ndarray) -> None:
        pass

    def get_metadata(self) -> dict:
        return self.metadata

    # TODO support masked dataset loading


class ENVISpectralIO(AbstractSpectralIO):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.metadata = self._validate_envi_header(self.file_path)

    # TODO implement __getitem__ to support slicing operations

    def get_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int]) -> np.ndarray:

        # Validates range coordinates
        x_range = (max(0, x_range[0]), min(self.metadata["samples"], x_range[1]))
        y_range = (max(0, y_range[0]), min(self.metadata["lines"], y_range[1]))
        z_range = (max(0, z_range[0]), min(self.metadata["bands"], z_range[1]))

        if self.metadata["byte order"] == 0:
            byte_order_flag = "<"
        else:
            byte_order_flag = ">"

        if self.metadata["data type"] == 1:
            num_bytes_per_data_point = 1
            format_character_flag = "B"
            data_type = np.uint8

        elif self.metadata["data type"] == 2:
            num_bytes_per_data_point = 2
            format_character_flag = "h"
            data_type = np.int16

        elif self.metadata["data type"] == 3:
            num_bytes_per_data_point = 4
            format_character_flag = "i"
            data_type = np.int32

        elif self.metadata["data type"] == 4:
            num_bytes_per_data_point = 4
            format_character_flag = "f"
            data_type = np.float32

        elif self.metadata["data type"] == 5:
            num_bytes_per_data_point = 8
            format_character_flag = "d"
            data_type = np.float64

        elif self.metadata["data type"] == 6:
            raise NotImplementedError("Error, this parser does not yet support complex valued data")  # TODO

        elif self.metadata["data type"] == 9:
            raise NotImplementedError("Error, this parser does not yet support complex valued data")  # TODO

        elif self.metadata["data type"] == 12:
            num_bytes_per_data_point = 2
            format_character_flag = "H"
            data_type = np.uint16

        elif self.metadata["data type"] == 13:
            num_bytes_per_data_point = 4
            format_character_flag = "I"
            data_type = np.uint32

        elif self.metadata["data type"] == 14:
            num_bytes_per_data_point = 8
            format_character_flag = "q"
            data_type = np.int64

        elif self.metadata["data type"] == 15:
            num_bytes_per_data_point = 8
            format_character_flag = "Q"
            data_type = np.uint64

        else:
            raise ValueError(f"Error: data type of value {self.metadata['data type']} un-parsable.")

        volume = np.zeros(shape=(y_range[1]-y_range[0],
                                 x_range[1]-x_range[0],
                                 z_range[1]-z_range[0]), dtype=data_type)

        # TODO, file open persistence using with keyword

        with open(self.file_path.rstrip(".hdr"), "rb") as infile:
            file_offset = (
                    self.metadata["header offset"]
                    + ((x_range[0])
                       + (z_range[0] * self.metadata["samples"])
                       + (y_range[0] * self.metadata["samples"] * self.metadata["bands"]))
                    * num_bytes_per_data_point)

            if self.metadata["interleave"] == "bil":
                for y in range(y_range[1] - y_range[0]):
                    for z in range(z_range[1] - z_range[0]):
                        infile.seek(file_offset, 1)

                        sample = infile.read(num_bytes_per_data_point * (x_range[1] - x_range[0]))

                        converted_sample = unpack_from(byte_order_flag
                                                       + (format_character_flag * (x_range[1] - x_range[0])),
                                                       sample)

                        volume[y, :, z] = converted_sample

                        file_offset = (self.metadata["samples"] - x_range[1] + x_range[0]) * num_bytes_per_data_point
                    file_offset += (self.metadata["samples"]
                                    * (self.metadata["bands"]
                                       - z_range[1]
                                       + z_range[0])) * num_bytes_per_data_point

            elif self.metadata["interleave"] == "bsq":
                raise NotImplementedError("Error, BSQ not yet supported")  # TODO

            elif self.metadata["interleave"] == "bip":
                raise NotImplementedError("Error, BIP not yet supported")  # TODO

            else:
                raise ValueError(f"Error, unable to read interleave type {self.metadata['interleave']}")

        return volume

    # TODO, implement file writing
    def set_volume_chunk(self,
                         x_range: Tuple[int, int],
                         y_range: Tuple[int, int],
                         z_range: Tuple[int, int],
                         data: np.ndarray) -> None:
        raise NotImplementedError

    @staticmethod
    def _parse_envi_header(header_path: str) -> dict:
        header_data = {}
        with open(header_path) as infile:
            data_string = infile.read()

        # Assure proper formatting and remove first line that always says ENVI
        assert data_string.startswith("ENVI\n")
        data_string = data_string[5:]

        # Separate comments from data
        comments = []
        for match in re.finditer(r";(.*)\n?", data_string):
            comments.append(match.group(1))
            data_string = data_string.replace(match.group(0), "", 1)

        if comments:
            header_data["comments"] = comments
            # TODO log that the file had comments and list them

        # Get the required ENVI file metadata
        required_expressions_tuple = [
            ("file type", r"(file type) = (.*)\n?", str),
            ("interleave", r"(interleave) = (bil|bsq|bip)\n?", str),
            ("data type", r"(data type) = (12|13|14|15|1|2|3|4|5|6|9|)\n?", int),
            ("bands", r"(bands) = (\d+)\n?", int),
            ("byte order", r"(byte order) = ([01])\n?", int),
            ("header offset", r"(header offset) = (\d+)\n?", int),
            ("lines", r"(lines) = (\d+)\n?", int),
            ("samples", r"(samples) = (\d+)\n?", int)
        ]
        for keyword, expression, casting_function in required_expressions_tuple:
            # TODO, find a clean way of checking for repeated keywords and warning/throwing an error if that occurs
            match = re.search(expression, data_string)
            if match:
                header_data[keyword] = casting_function(match.group(2))
                data_string = data_string[:match.start()] + data_string[match.end():]
            else:
                raise ValueError(f"Error, the ENVI header is missing the required information {keyword}")

        extra_data = {}
        while len(data_string) > 0:
            match = re.match(r"(.*) = ({(.|\n)*?})\n?", data_string)
            if match:
                extra_data[match.group(1)] = match.group(2)
                data_string = data_string[:match.start()] + data_string[match.end():]
            else:
                match = re.match(r"(.*) = (.*)\n?", data_string)
                if match:
                    extra_data[match.group(1)] = match.group(2)
                    data_string = data_string[:match.start()] + data_string[match.end():]
                else:
                    raise ValueError(f"Can not parse data left in ENVI file: {data_string}")

        if extra_data:
            # TODO, log extra data
            header_data["extra data"] = extra_data

        return header_data

    def _validate_envi_header(self, header_path: str) -> dict:
        try:
            return self._parse_envi_header(header_path)

        except Exception as error:
            print(f"Error, failed to parse ENVI header file {header_path}")
            raise error


class SpectralDataHandler:
    def __init__(self, file_path: str, file_format: str = None):
        self.file_path = file_path

        if file_format:
            self.file_format = file_format
        else:
            self.file_format = self._infer_file_format(self.file_path)

        if self.file_format == "envi":
            self.io = ENVISpectralIO(self.file_path)
            self.metadata = self.io.get_metadata()

        elif self.file_format == "path_ref":
            with open(self.file_path) as infile:
                reference_path = infile.read()

            dummy_handler = SpectralDataHandler(reference_path)
            self.io = dummy_handler.io
            self.metadata = dummy_handler.io.get_metadata()

        else:
            # TODO
            raise NotImplementedError(f"Error, file handling for file format {self.file_format} not yet supported")

    def _infer_file_format(self, file_path: str) -> str:
        if file_path.endswith((".hdf", ".hdf5", ".h5", "he5")):
            return "hdf"
        elif file_path.endswith((".fits", ".fit", ".fts")):
            return "fits"
        elif file_path.endswith(".hdr") and os.path.exists(file_path[:-4]):
            return "envi"
        elif "." not in file_path and os.path.exists(file_path + ".hdr"):
            self.file_path = file_path + ".hdr"  # TODO, not sure if we should allow inputting data file or header
            return "envi"
        elif file_path.endswith("path_ref"):
            return "path_ref"
        elif os.path.isdir(file_path):
            # TODO, check that format matches spectral camera output
            pass
        else:
            raise ValueError(f"Error, could not infer file format for file {file_path}. Is it supported?")


class SpectralPackageManager:
    @property
    def sample_ids(self):
        return [directory for directory in os.listdir(self.samples_path)
                if os.path.isdir(os.path.join(self.samples_path, directory))]

    @property
    def transform_ids(self):
        return [directory for directory in os.listdir(self.transforms_path) if os.path.isdir(directory)]

    @property
    def pipeline_ids(self):
        return [directory for directory in os.listdir(self.pipelines_path) if os.path.isdir(directory)]

    @classmethod
    def make_package(cls, base_path, project_id):
        file_path = os.path.join(base_path, project_id + ".spctrl")
        os.makedirs(file_path)

        for top_level_folder in ("samples", "transforms", "pipelines", "pipeline_output"):
            os.makedirs(os.path.join(file_path, top_level_folder))

        # TODO, in this stage, add a metadata json file specifying package version so
        #  hierarchy can be changed in the future if needed
        manager = cls(file_path)
        manager.package_logger.info("SpectralPkg named {} created at {}".format(project_id + '.spctrl', base_path))

        return manager

    def __init__(self, project_path):
        self.filepath = project_path
        self.samples_path = os.path.join(self.filepath, "samples")
        self.transforms_path = os.path.join(self.filepath, "transforms")
        self.pipelines_path = os.path.join(self.filepath, "pipelines")

        if os.path.exists(self.filepath):
            # TODO, add a formatter to each file handler so it lists time and user for event listing
            self.package_logger = logging.getLogger("package")
            self.package_logger.setLevel(logging.INFO)
            self.package_logger.addHandler(logging.FileHandler(os.path.join(self.filepath, "package_log.log")))

            self.samples_logger = logging.getLogger("samples")
            self.samples_logger.addHandler(logging.FileHandler(os.path.join(self.samples_path, "samples_log.log")))

            self.transforms_logger = logging.getLogger("transforms")
            self.transforms_logger.addHandler(logging.FileHandler(os.path.join(self.transforms_path, "transforms_log.log")))

            self.pipelines_logger = logging.getLogger("pipelines")
            self.pipelines_logger.addHandler(logging.FileHandler(os.path.join(self.pipelines_path, "pipelines_log.log")))

        else:
            raise ValueError("Error, path to spectral package does not exist")

    def add_sample(self, path, sample_id=None, overwrite=False, copy_path=False, cont_mask=None, class_mask=None):
        if os.path.isdir(path):
            if sample_id is None:
                sample_id = os.path.basename(os.path.normpath(path))
            file_type = None

        else:
            base, file_type = os.path.basename(os.path.normpath(path)).split(".")
            if sample_id is None:
                sample_id = base

        if copy_path:
            file_type = "path_ref"

        destination = os.path.join(self.filepath, "samples", sample_id)

        # Check to see if we would be overwriting existing data, handle by set preference
        if os.path.exists(destination):
            if not overwrite:
                self.samples_logger.warning('Tried to write file {} to {}. Process stopped due to existing '
                                            'file at that location'.format(path, destination))
                raise ValueError("Error, artifact with identifier {} already exists".format(sample_id))
            else:
                self.samples_logger.warning('Overwriting existing file at {}'.format(destination))
                shutil.rmtree(destination)

        os.makedirs(destination, exist_ok=True)
        if file_type == "":
            file_destination = os.path.join(destination, "spectral_data")
        else:
            file_destination = os.path.join(destination, "spectral_data.{}".format(file_type))

        if file_type == "path_ref":
            with open(file_destination, "w") as outfile:
                outfile.write(path)
        elif file_type == "":
            shutil.copy2(path, file_destination)
            shutil.copy2(path + ".hdr", file_destination + ".hdr")
        elif file_type == "hdr":
            shutil.copy2(path, file_destination)
            shutil.copy2(path[:-4], file_destination[:-4])
        elif file_type is None:
            shutil.copytree(path, file_destination)
        else:
            shutil.copy2(path, file_destination)

        self.samples_logger.info(f"Copied new sample with id {sample_id} from {path} to {file_destination}")

        # TODO, handle addition of masks with unique identifiers, reconcile into a single mask system
        if cont_mask is not None:
            pass
        if class_mask is not None:
            pass

        # TODO, also write metadata json file once you know what metadata needs stored

    def remove_sample(self, identifier):
        sample_path = os.path.join(self.filepath, "samples", identifier)
        shutil.rmtree(sample_path)

        self.samples_logger.info(f"Removed sample at location {sample_path}")

    def get_sample_handler(self, sample_id):
        data_dir = os.path.join(self.samples_path, sample_id)
        data_file = [os.path.join(data_dir, file)
                     for file in os.listdir(data_dir) if "spectral_data" in file][0]
        return SpectralDataHandler(data_file)
