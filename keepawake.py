"""Keep Windows out of Modern Standby for as long as the calling process lives.

This is not a nicety. On a Modern Standby machine the system drops into low-power idle a
minute or so after the user walks away, even with work running: the Desktop Activity
Moderator suspends background Win32 processes, and they are torn down when the system comes
back out. Three separate overnight runs of this pipeline were lost that way. The System
event log records it as Kernel-Power 506 going in and 507 coming out; the giveaway is that
the run's own log file gains nothing at all between those two timestamps.

`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` registers a system power
request that blocks the idle transition. It is the mechanism installers and media players
use, it changes none of the user's power settings, and it lapses automatically as soon as
the process exits -- including when it is killed -- so the machine goes back to sleeping
normally the moment the run is over.

Two things it deliberately does not do: it does not keep the display on (ES_DISPLAY_REQUIRED
is not set, so the screen still blanks and the machine still locks), and it does not override
a sleep the user asks for explicitly by closing the lid or choosing Sleep.

    import keepawake; keepawake.hold("the CIFAR stage")
"""
import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def hold(reason=""):
    """Ask Windows not to enter standby while this process runs. True if it took."""
    if not sys.platform.startswith("win"):
        return False
    try:
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as e:
        print(f"could not request that the system stay awake: {e}", flush=True)
        return False
    if prev == 0:
        print("the request that the system stay awake was refused", flush=True)
        return False
    print(f"holding the system awake{' for ' + reason if reason else ''} "
          f"(released automatically when this process exits)", flush=True)
    return True


def release():
    """Drop the request. Not usually needed: process exit drops it anyway."""
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


if __name__ == "__main__":
    import time
    hold("a manual hold; press Ctrl+C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        release()
