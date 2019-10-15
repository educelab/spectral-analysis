import os
import re


class SpectralDataHandler:
    file_path: str = None
    file_format: str = None
    metadata: dict = {}

    def __init__(self, file_path: str = None, file_format: str = None):
        if file_path:
            self.file_path = file_path

        if file_format:
            self.file_format = file_format
        else:
            self.file_format = self._infer_file_format(self.file_path)

        if self.file_format == "envi":
            self.metadata = self._validate_envi_header(self.file_path)

        # TODO, add other file_format metadata parsing

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
        elif os.path.isdir(file_path):
            # TODO, check that format matches spectral camera output
            pass
        else:
            raise ValueError(f"Error, could not infer file format for file {file_path}")

    @ staticmethod
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
                    raise ValueError(f"Unparsable data left in ENVI file: {data_string}")

        if extra_data:
            # TODO, log extra data
            header_data["extra data"] = extra_data

        return header_data

    def _validate_envi_header(self, header_path: str) -> dict:
        try:
            return self._parse_envi_header(header_path)
        except ValueError:
            raise ValueError(f"Error, failed to parse ENVI header file {header_path}")


if __name__ == "__main__":
    # TODO write argument parser for file io
    x = SpectralDataHandler("/Volumes/HD-Daniel/PHerc118/Photos/2017-Hyperspectral/RawScans/PHerc118-Pezzo1/2017_07_17_10_25_13/2017_07_17_10_25_13raw.hdr")
