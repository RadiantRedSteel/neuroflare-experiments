import nidaqmx
from nidaqmx.constants import AcquisitionType

def pulse_multiple_counters(ctr_indices, duration_seconds=0.01):
    """
    Allows pulsing multiple counters simultaneously.

    Args:
        ctr_indices (list of int): List of counter indices (0, 1, 2, 3) to pulse.
        duration_seconds (float): Duration of the pulse in seconds.
    """
    # Mapping of counter index to its default/desired PFI terminal on USB-6363 BNC
    # CTR 0 -> PFI 0 (TriggerBox EMG)
    # CTR 1 -> PFI 1 (EMG Start)
    # CTR 2 -> PFI 2 (EMG Stop)
    # CTR 3 -> PFI 3 (Digitimer)
    # CTR 4 -> PFI 4 (TriggerBox Digitimer)
    # Note: On USB-6363 BNC, PFI 0..7 are available on the BNC connectors.

    with nidaqmx.Task() as task:
        for idx in ctr_indices:
            ctr_name = f"Dev1/ctr{idx}"
            pfi_term = f"/Dev1/PFI{idx}"

            chan = task.co_channels.add_co_pulse_chan_time(
                ctr_name,
                high_time=duration_seconds,
                low_time=1e-7,       # 100 ns (safely above the ~20ns timebase limit)
                initial_delay=1e-7   # 100 ns delay before start
            )
            chan.co_pulse_term = pfi_term

        # Implicit timing for all channels in the task
        task.timing.cfg_implicit_timing(
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=1
        )

        task.start()
        # When starting a task with multiple channels, NI-DAQmx starts them sequentially in software.
        # This results in a very small jitter (likely microseconds). For many applications, this is
        # perfectly acceptable and counts as "simultaneous" for human-perceived events.
        # If you need nanosecond-level hardware synchronization, you would configure them to
        # wait for a shared 'Arm Start Trigger'.
        task.wait_until_done(timeout=1.0)