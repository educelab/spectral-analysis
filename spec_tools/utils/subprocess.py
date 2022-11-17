import logging
import subprocess
import sys
from typing import List


def run_command(cmd: List[str], cwd=None):
    logger = logging.getLogger(__name__)
    """Run a command as a subprocess"""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except OSError as e:
        logger.error(f'Failed to start command: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except subprocess.SubprocessError as e:
        logger.error(f'Command failed: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except:
        sys.exit(f'Unexpected error: {sys.exc_info()[0]}')
