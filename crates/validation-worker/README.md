# DiariZen validation worker

The worker accepts a frozen development bundle as a plain tar file or a Zstandard-compressed tar file. The tar file can use USTAR, plain GNU, or legacy headers. It can contain only regular files and zero-size directories. It must not contain links, devices, sparse entries, or GNU or PAX extension records.

The expanded bundle limits are:

- 16 GiB total regular-file data
- 20,000 entries
- 4 GiB for one regular file
- 512 UTF-8 bytes for one relative path

Use this Python recipe to create a supported tar file:

```python
from pathlib import Path
import tarfile

source = Path("frozen-dev")


def supported(info: tarfile.TarInfo) -> tarfile.TarInfo:
    if not (info.isfile() or info.isdir()):
        raise ValueError(f"unsupported bundle source type: {info.name}")
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


with tarfile.open("dev-bundle.tar", "w", format=tarfile.USTAR_FORMAT) as archive:
    archive.add(source, arcname=".", recursive=True, filter=supported)
```

The source root must contain `bundle.json`. Compress `dev-bundle.tar` with Zstandard when the artifact declaration uses `application/zstd` and `compression: "zstd"`.
