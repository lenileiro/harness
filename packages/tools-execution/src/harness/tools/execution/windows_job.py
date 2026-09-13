"""Kill-on-close Windows Job Object ownership, imported safely on any host.

https://learn.microsoft.com/windows/win32/procthread/job-objects
No breakaway flags are enabled. This manages ordinary CreateProcess descendants;
it is not a sandbox for hostile code using out-of-process system services.
"""

from __future__ import annotations

import ctypes
from typing import Any


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32() -> Any:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("Windows Job Objects require a native Windows host")
    api = loader("kernel32", use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    api.CreateJobObjectW.restype = ctypes.c_void_p
    api.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    api.SetInformationJobObject.restype = ctypes.c_int
    api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenProcess.restype = ctypes.c_void_p
    api.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    api.AssignProcessToJobObject.restype = ctypes.c_int
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    return api


class WindowsJob:
    def __init__(self) -> None:
        self.api = _kernel32()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("could not create Windows Job Object")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise OSError("could not enable Windows Job Object kill-on-close ownership")

    def assign(self, pid: int) -> None:
        process = self.api.OpenProcess(0x100 | 0x1, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise OSError("could not open worker for Windows Job Object assignment")
        try:
            if not self.api.AssignProcessToJobObject(self.handle, process):
                raise OSError("could not assign worker to Windows Job Object; execution refused")
        finally:
            self.api.CloseHandle(process)

    def close(self) -> None:
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise OSError("could not close Windows Job Object")
